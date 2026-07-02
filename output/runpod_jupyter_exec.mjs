#!/usr/bin/env node

import { randomUUID } from 'node:crypto';

const base = process.env.RUNPOD_JUPYTER_BASE || '';
const token = process.env.RUNPOD_JUPYTER_TOKEN || '';
const kernelName = process.env.RUNPOD_JUPYTER_KERNEL || 'python3';
const codeArg = process.argv.slice(2).join(' ').trim();

async function readStdin() {
  if (process.stdin.isTTY) return '';
  return await new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (chunk) => {
      data += chunk;
    });
    process.stdin.on('end', () => resolve(data));
    process.stdin.on('error', reject);
  });
}

function buildMessage({ session, msgType, content }) {
  const msgId = randomUUID();
  return JSON.stringify({
    header: {
      msg_id: msgId,
      username: 'codex',
      session,
      msg_type: msgType,
      version: '5.3',
      date: new Date().toISOString(),
    },
    parent_header: {},
    metadata: {},
    channel: 'shell',
    content,
    buffers: [],
  });
}

async function createKernel() {
  const response = await fetch(`${base}/api/kernels?token=${encodeURIComponent(token)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: kernelName }),
  });
  if (!response.ok) {
    throw new Error(`Kernel create failed: ${response.status} ${await response.text()}`);
  }
  return await response.json();
}

async function deleteKernel(kernelId) {
  const response = await fetch(`${base}/api/kernels/${kernelId}?token=${encodeURIComponent(token)}`, {
    method: 'DELETE',
  });
  if (!response.ok && response.status !== 404) {
    throw new Error(`Kernel delete failed: ${response.status} ${await response.text()}`);
  }
}

async function executeCode(kernelId, code) {
  const session = randomUUID();
  const requestId = randomUUID();
  const wsBase = base.replace(/^http/, 'ws');
  const wsUrl = `${wsBase}/api/kernels/${kernelId}/channels?token=${encodeURIComponent(token)}`;
  const ws = new WebSocket(wsUrl);

  return await new Promise((resolve, reject) => {
    const stdout = [];
    const stderr = [];
    const rich = [];
    let gotReply = false;
    let gotIdle = false;
    let done = false;

    const finish = (result) => {
      if (done) return;
      done = true;
      try {
        ws.close();
      } catch {}
      resolve(result);
    };

    const fail = (error) => {
      if (done) return;
      done = true;
      try {
        ws.close();
      } catch {}
      reject(error);
    };

    ws.addEventListener('open', () => {
      ws.send(
        JSON.stringify({
          header: {
            msg_id: requestId,
            username: 'codex',
            session,
            msg_type: 'execute_request',
            version: '5.3',
            date: new Date().toISOString(),
          },
          parent_header: {},
          metadata: {},
          channel: 'shell',
          session,
          content: {
            code,
            silent: false,
            store_history: false,
            user_expressions: {},
            allow_stdin: false,
            stop_on_error: true,
          },
          buffers: [],
        }),
      );
    });

    ws.addEventListener('message', (event) => {
      const raw = typeof event.data === 'string' ? event.data : Buffer.from(event.data).toString('utf8');
      const msg = JSON.parse(raw);
      const msgType = msg?.msg_type || msg?.header?.msg_type;
      const content = msg?.content || {};
      const parentId = msg?.parent_header?.msg_id;

      if (parentId && parentId !== requestId) {
        return;
      }

      if (msgType === 'stream') {
        if (content.name === 'stderr') stderr.push(content.text || '');
        else stdout.push(content.text || '');
        return;
      }

      if (msgType === 'error') {
        const traceback = Array.isArray(content.traceback) ? content.traceback.join('\n') : '';
        fail(new Error(traceback || `${content.ename || 'Error'}: ${content.evalue || 'unknown'}`));
        return;
      }

      if (msgType === 'execute_result' || msgType === 'display_data') {
        const data = content.data || {};
        if (typeof data['text/plain'] === 'string') rich.push(data['text/plain']);
        return;
      }

      if (msgType === 'execute_reply') {
        gotReply = true;
        if (content.status === 'error') {
          fail(new Error(`${content.ename || 'Error'}: ${content.evalue || 'unknown'}`));
          return;
        }
        if (gotIdle) finish({ stdout: stdout.join(''), stderr: stderr.join(''), rich: rich.join('\n') });
        return;
      }

      if (msgType === 'status' && content.execution_state === 'idle') {
        gotIdle = true;
        if (gotReply) finish({ stdout: stdout.join(''), stderr: stderr.join(''), rich: rich.join('\n') });
      }
    });

    ws.addEventListener('error', (event) => {
      fail(new Error(`WebSocket error: ${event.message || 'unknown'}`));
    });

    ws.addEventListener('close', () => {
      if (!done && !gotReply) {
        fail(new Error('Kernel channel closed before execution completed'));
      }
    });
  });
}

const stdinCode = await readStdin();
const code = codeArg || stdinCode.trim();
if (!code) {
  console.error('Pass code as an argument or pipe it on stdin.');
  process.exit(2);
}
if (!base || !token) {
  console.error('Set RUNPOD_JUPYTER_BASE and RUNPOD_JUPYTER_TOKEN before using this helper.');
  process.exit(2);
}

let kernel;
try {
  kernel = await createKernel();
  const result = await executeCode(kernel.id, code);
  if (result.stdout) process.stdout.write(result.stdout);
  if (result.rich) {
    if (result.stdout && !result.stdout.endsWith('\n')) process.stdout.write('\n');
    process.stdout.write(result.rich);
    if (!result.rich.endsWith('\n')) process.stdout.write('\n');
  }
  if (result.stderr) process.stderr.write(result.stderr);
} finally {
  if (kernel?.id) {
    await deleteKernel(kernel.id).catch((error) => {
      console.error(String(error));
    });
  }
}
