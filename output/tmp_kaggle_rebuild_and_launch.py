from pathlib import Path
import zipfile

DATASET_FILE_NAMES = [
    'conversation_data.quality_anchor_v2.jsonl',
    'conversation_data.coding_knowledge_2026_02_19.jsonl',
    'conversation_data.world_events_2026_02_19.jsonl',
    'conversation_data.supermix_plus_v27_500k.jsonl',
    'conversation_data.mega_reasoning_creative_v25_75582.jsonl',
    'conversation_data.mega_creative_250k_v2.jsonl',
]

def extract_training_archives_recursive() -> None:
    if not TRAINING_INPUT_DIR.exists():
        return
    dest_dir = WORKSPACE_DIR / 'datasets'
    dest_dir.mkdir(parents=True, exist_ok=True)
    for archive_path in sorted(TRAINING_INPUT_DIR.rglob('*.zip')):
        print(f'Extracting training archive: {archive_path} -> {dest_dir}')
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)

def resolve_dataset_files_recursive() -> list[str]:
    resolved: list[str] = []
    missing: list[str] = []
    search_roots = [
        TRAINING_INPUT_DIR,
        TRAINING_INPUT_DIR / 'datasets',
        WORKSPACE_DIR / 'datasets',
        WORKSPACE_DIR,
        KAGGLE_WORKING_ROOT,
    ]
    for name in DATASET_FILE_NAMES:
        candidates: list[Path] = []
        for root in search_roots:
            if not root.exists():
                continue
            direct = [root / name, root / 'datasets' / name]
            for candidate in direct:
                if candidate.exists() and candidate.is_file():
                    candidates.append(candidate)
            if not candidates:
                try:
                    for hit in sorted(root.rglob(name)):
                        if hit.is_file():
                            candidates.append(hit)
                except Exception:
                    pass
        chosen = next((candidate for candidate in candidates if candidate.exists()), None)
        if chosen is None:
            missing.append(name)
        else:
            resolved.append(str(chosen.resolve()))
    if missing:
        raise FileNotFoundError('Missing training JSONL files: ' + ', '.join(missing))
    return resolved

extract_training_archives_recursive()
DATA_FILES = resolve_dataset_files_recursive()
print('Resolved dataset files:')
for dataset_path in DATA_FILES:
    print('-', dataset_path)

import json
import os
import shlex
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone

LOGICAL_CPU = max(1, os.cpu_count() or 1)
INTEROP_CPU = max(1, min(4, LOGICAL_CPU // 2))


def get_latest_adapter_checkpoint(run_dir: Path | None) -> Path | None:
    if run_dir is None or not run_dir.exists():
        return None
    direct_adapter = run_dir / 'adapter'
    if (direct_adapter / 'adapter_config.json').exists() and (direct_adapter / 'adapter_model.safetensors').exists():
        return direct_adapter.resolve()
    if (run_dir / 'adapter_config.json').exists() and (run_dir / 'adapter_model.safetensors').exists():
        return run_dir.resolve()
    latest_file = run_dir / 'latest_adapter_checkpoint.txt'
    if latest_file.exists():
        checkpoint_dir = latest_file.read_text(encoding='utf-8').strip()
        if checkpoint_dir:
            raw = Path(checkpoint_dir)
            candidates = [raw]
            if not raw.is_absolute():
                candidates.append(WORKSPACE_DIR / raw)
                candidates.append(run_dir / raw)
            for candidate in candidates:
                if candidate.exists():
                    return candidate.resolve()
    checkpoints_dir = run_dir / 'checkpoints'
    if not checkpoints_dir.exists():
        return None
    metas = sorted(checkpoints_dir.rglob('checkpoint_meta.json'), key=lambda path: path.stat().st_mtime, reverse=True)
    for meta in metas:
        adapter_dir = meta.parent / 'adapter'
        if adapter_dir.exists():
            return adapter_dir.resolve()
    return None


def extract_warm_start_archives() -> Path | None:
    if not WARM_START_INPUT_DIR.exists():
        return None
    archive_paths = sorted(WARM_START_INPUT_DIR.glob('*.zip'))
    if not archive_paths:
        return None
    extracted_dir = KAGGLE_WORKING_ROOT / 'attached_warm_start_input'
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    for archive_path in archive_paths:
        print(f'Extracting warm-start archive: {archive_path} -> {extracted_dir}')
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extracted_dir)
    return extracted_dir


def find_attached_warm_start_dir() -> Path | None:
    candidates = [
        WARM_START_INPUT_DIR,
        WARM_START_INPUT_DIR / 'artifacts',
    ]
    for candidate in candidates:
        if candidate.exists() and get_latest_adapter_checkpoint(candidate) is not None:
            return candidate
    if WARM_START_INPUT_DIR.exists():
        for child in sorted(WARM_START_INPUT_DIR.iterdir()):
            if child.is_dir() and get_latest_adapter_checkpoint(child) is not None:
                return child
    extracted_dir = extract_warm_start_archives()
    if extracted_dir is not None and get_latest_adapter_checkpoint(extracted_dir) is not None:
        return extracted_dir
    if extracted_dir is not None:
        for child in sorted(extracted_dir.iterdir()):
            if child.is_dir() and get_latest_adapter_checkpoint(child) is not None:
                return child
    return None


INPUT_WARM_START_DIR = find_attached_warm_start_dir()


def seed_working_warm_start() -> Path | None:
    existing_checkpoint = get_latest_adapter_checkpoint(RESUME_WARM_START_DIR)
    if existing_checkpoint is not None:
        return existing_checkpoint
    input_checkpoint = get_latest_adapter_checkpoint(INPUT_WARM_START_DIR)
    if input_checkpoint is None or INPUT_WARM_START_DIR is None:
        return None
    print(f'Seeding writable warm-start artifact from Kaggle input: {INPUT_WARM_START_DIR}')
    shutil.copytree(INPUT_WARM_START_DIR, RESUME_WARM_START_DIR, dirs_exist_ok=True)
    return get_latest_adapter_checkpoint(RESUME_WARM_START_DIR)


def append_scalar_arg(cmd: list[str], flag: str, value: object) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def append_bool_flag(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def detect_gpu_profile() -> tuple[str, str, int]:
    query = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
        text=True,
    ).splitlines()[0]
    gpu_name_raw, memory_mb_raw = [part.strip() for part in query.split(',', 1)]
    gpu_name = gpu_name_raw.upper()
    memory_mb = int(float(memory_mb_raw))
    forced = str(GPU_PROFILE).strip().lower()
    if forced and forced != 'auto':
        return forced, gpu_name_raw, memory_mb
    if 'A100' in gpu_name:
        return 'a100', gpu_name_raw, memory_mb
    if 'L4' in gpu_name:
        return 'l4', gpu_name_raw, memory_mb
    if 'P100' in gpu_name:
        return 'p100', gpu_name_raw, memory_mb
    if 'T4' in gpu_name:
        return 't4', gpu_name_raw, memory_mb
    return 'generic', gpu_name_raw, memory_mb


def extract_teacher_archives() -> list[Path]:
    extracted: list[Path] = []
    if not TEACHER_INPUT_DIR.exists():
        return extracted
    dest_dir = WORKSPACE_DIR / 'runtime_python'
    for archive_path in sorted(TEACHER_INPUT_DIR.glob('*.zip')):
        print(f'Extracting teacher archive: {archive_path} -> {dest_dir}')
        dest_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
        extracted.append(archive_path)
    return extracted


def resolve_teacher_asset(filename: str) -> Path | None:
    candidates = [
        TEACHER_INPUT_DIR / filename,
        TEACHER_INPUT_DIR / 'runtime_python' / filename,
        WORKSPACE_DIR / 'runtime_python' / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


TEACHER_ARCHIVES = extract_teacher_archives()
if TEACHER_ARCHIVES:
    print('Attached teacher archive(s):')
    for archive_path in TEACHER_ARCHIVES:
        print('-', archive_path)

gpu_profile, detected_gpu_name, detected_gpu_memory_mb = detect_gpu_profile()
print(f'GPU profile: {gpu_profile} ({detected_gpu_name}, {detected_gpu_memory_mb} MiB)')

base_scalar_args = {
    '--max_records': 600000,
    '--max_source_fraction': 0.52,
    '--max_synthetic_fraction': 0.06,
    '--max_prompt_signature_count': 4,
    '--data_log_every_records': 2000,
    '--prompt_signature_cap_exempt_sources': 'conversation_data.quality_anchor_v2.jsonl,conversation_data.mega_reasoning_creative_v25_75582.jsonl',
    '--eval_size': 500,
    '--eval_split_mode': 'auto',
    '--eval_min_quality_score': 1.05,
    '--max_length': 448,
    '--batch_size': 1,
    '--grad_accum_steps': 16,
    '--epochs': 6,
    '--max_steps': 6200,
    '--lr': '1.0e-5',
    '--sft_lr_schedule': 'cosine_restarts',
    '--sft_lr_restart_period': 620,
    '--sft_warmup_steps': 30,
    '--sft_min_lr_ratio': 0.22,
    '--sft_max_grad_norm': 0.9,
    '--sft_focal_gamma': 1.35,
    '--sft_eval_every_steps': 240,
    '--sft_early_stop_patience': 5,
    '--sft_curriculum_quality_ramp': 0.22,
    '--sft_grad_noise_eta': 0.01,
    '--train_log_every_steps': 1,
    '--save_every_steps': SAVE_EVERY_STEPS,
    '--weight_decay': 0.02,
    '--lora_r': 32,
    '--lora_alpha': 64,
    '--lora_dropout': 0.03,
    '--lora_init': 'pissa_niter_4',
    '--lora_plus_ratio': 16,
    '--neftune_noise_alpha': 5.0,
    '--sft_weight_mode': 'quality',
    '--sft_min_weight': 0.62,
    '--sft_max_weight': 1.88,
    '--sft_synthetic_prompt_weight': 0.62,
    '--sft_teacher_source_weight': 0.92,
    '--sft_quality_anchor_boost': 1.14,
    '--sft_coding_boost': 1.24,
    '--sft_events_boost': 1.08,
    '--sft_reasoning_boost': 1.28,
    '--sft_prompt_skill_boost': 1.17,
    '--sft_conversation_boost': 1.24,
    '--sft_creativity_boost': 1.16,
    '--sft_knowledge_density_boost': 1.22,
    '--sft_rdrop_alpha': 0.05,
    '--sft_length_bucket_window_mult': 24,
    '--sft_followup_paraphrase_aug': 1,
    '--sft_followup_paraphrase_weight': 0.68,
    '--sft_min_quality_score': 0.98,
    '--sft_quality_filter_exempt_sources': 'conversation_data.quality_anchor_v2.jsonl,conversation_data.world_events_2026_02_19.jsonl',
    '--sft_source_balance_strength': 0.66,
    '--sft_source_balance_max_scale': 1.95,
    '--sft_packing_max_samples_per_row': 3,
    '--sft_selection_strategy': 'none',
    '--sft_selection_keep_ratio': 1.0,
    '--sft_selection_min_keep': 0,
    '--sft_selection_max_keep': 0,
    '--sft_selection_hardness_target': 0.45,
    '--sft_selection_hardness_bandwidth': 0.20,
    '--sft_selection_budget_mode': 'tokens',
    '--sft_selection_budget_power': 0.5,
    '--sft_selection_scope': 'all',
    '--sft_selection_scope_min_words': 8,
    '--preference_objective': 'ipo',
    '--preference_steps': 1500,
    '--preference_rescore_every': 25,
    '--preference_pairs': 34000,
    '--preference_candidate_count': 8,
    '--preference_reject_similarity_min': 0.16,
    '--preference_beta': 1.9,
    '--preference_beta_end': 3.6,
    '--preference_margin': 0.00,
    '--preference_margin_end': 0.00,
    '--preference_label_smoothing': 0.03,
    '--preference_sft_weight': 0.32,
    '--preference_length_weight': 0.08,
    '--preference_length_control_strength': 0.0,
    '--preference_length_control_target_ratio': 1.0,
    '--preference_length_control_max_penalty': 0.0,
    '--preference_hardness_gamma': 1.15,
    '--preference_robust_alpha': 0.30,
    '--preference_robust_eta': 0.08,
    '--preference_robust_clip': 2.5,
    '--preference_wpo_alpha': 0.35,
    '--preference_wpo_clip': 2.5,
    '--preference_reference_anchor_weight': 0.04,
    '--preference_reference_anchor_batch_size': 2,
    '--preference_short_reject_boost': 0.75,
    '--preference_long_reject_boost': 0.25,
    '--preference_min_chosen_quality': 0.92,
    '--preference_min_chosen_words': 8,
    '--preference_min_quality_gap': 0.05,
    '--preference_max_pairs_per_user': 2,
    '--preference_max_pairs_per_source': 360,
    '--preference_mining_mode': 'auto',
    '--preference_mining_progress_every': 30,
    '--preference_mining_max_seconds': 4500,
    '--preference_mining_max_attempt_factor': 20,
    '--preference_coding_focus_boost': 1.30,
    '--preference_reasoning_focus_boost': 1.32,
    '--preference_counterfactual_rejects_per_prompt': 4,
    '--preference_selection_strategy': 'innovation_mix',
    '--preference_selection_keep_ratio': 0.62,
    '--preference_selection_min_keep': 1800,
    '--preference_selection_max_keep': 2400,
    '--preference_selection_hardness_target': 0.46,
    '--preference_selection_hardness_bandwidth': 0.22,
    '--preference_length_bucket_window_mult': 24,
    '--preference_lr': '1.4e-5',
    '--preference_lr_schedule': 'cosine',
    '--preference_warmup_steps': 18,
    '--preference_min_lr_ratio': 0.30,
    '--preference_max_grad_norm': 0.9,
    '--preference_max_new_tokens': 112,
    '--preference_prompt_max_tokens': 352,
    '--preference_self_play_budget': 0,
    '--preference_self_play_curriculum': 'easy_to_hard',
    '--preference_self_play_max_new_tokens': 0,
    '--supermix_distill_ratio': 0.14,
    '--supermix_distill_max': 8000,
    '--supermix_distill_best_of': 3,
    '--supermix_distill_log_every': 40,
    '--supermix_distill_max_seconds': 12000,
    '--supermix_distill_min_quality': 0.93,
    '--supermix_distill_min_gain': 0.18,
    '--supermix_distill_density_bias': 0.20,
    '--supermix_distill_gain_bias': 0.0,
    '--supermix_distill_compactness_bias': 0.0,
    '--supermix_distill_rank_margin': 0.0,
    '--seed': 48,
    '--device_preference': 'cuda,npu,xpu,mps,cpu,dml',
    '--model_dtype': 'auto',
    '--torch_num_threads': LOGICAL_CPU,
    '--torch_interop_threads': INTEROP_CPU,
}

gpu_profile_scalar_overrides = {
    't4': {
        '--max_length': 384,
        '--grad_accum_steps': 24,
        '--preference_pairs': 24000,
        '--preference_candidate_count': 6,
        '--supermix_distill_max': 5000,
        '--sft_packing_max_samples_per_row': 2,
    },
    'p100': {
        '--max_length': 384,
        '--grad_accum_steps': 28,
        '--preference_pairs': 22000,
        '--preference_candidate_count': 6,
        '--supermix_distill_max': 4000,
        '--sft_packing_max_samples_per_row': 2,
    },
    'l4': {
        '--max_length': 448,
        '--grad_accum_steps': 16,
        '--preference_pairs': 34000,
        '--preference_candidate_count': 8,
        '--supermix_distill_max': 8000,
        '--sft_packing_max_samples_per_row': 3,
    },
    'a100': {
        '--max_length': 512,
        '--grad_accum_steps': 12,
        '--preference_pairs': 42000,
        '--preference_candidate_count': 10,
        '--supermix_distill_max': 12000,
        '--sft_packing_max_samples_per_row': 4,
    },
    'generic': {},
}
profile_scalar_overrides = gpu_profile_scalar_overrides.get(gpu_profile, {})
if profile_scalar_overrides:
    print('Applying GPU-profile scalar overrides:', profile_scalar_overrides)
    base_scalar_args.update(profile_scalar_overrides)

teacher_weights_path = resolve_teacher_asset('champion_model_chat_supermix_v27_500k_ft.pth')
teacher_meta_path = resolve_teacher_asset('chat_model_meta_supermix_v27_500k.json')
if teacher_weights_path is None or teacher_meta_path is None:
    if float(base_scalar_args.get('--supermix_distill_ratio', 0.0) or 0.0) > 0.0:
        print('Supermix teacher assets not found; disabling teacher distillation for this Kaggle run.')
    teacher_weights_path = None
    teacher_meta_path = None
    base_scalar_args['--supermix_distill_ratio'] = 0.0
else:
    print(f'Using Supermix teacher weights: {teacher_weights_path}')
    print(f'Using Supermix teacher meta: {teacher_meta_path}')

base_bool_flags = {
    '--eval_drop_synthetic_prompts': True,
    '--use_rslora': True,
    '--use_dora': True,
    '--sft_length_bucketed_batches': True,
    '--sft_drop_synthetic_prompts': True,
    '--sft_auto_balance_sources': True,
    '--sft_true_packing': TRAIN_PROFILE == 'current_kaggle',
    '--preference_allow_template_prompts': True,
    '--preference_length_bucketed_batches': True,
    '--gradient_checkpointing': True,
}

optional_bool_flags = {
    '--supermix_distill_allow_synthetic_prompts': False,
}


def build_train_command() -> list[str]:
    cmd = [
        sys.executable,
        '-u',
        'source/qwen_supermix_pipeline.py',
        '--data',
        *DATA_FILES,
        '--base_model', BASE_MODEL,
        '--output_dir', str(OUTPUT_DIR),
        '--device', DEVICE,
    ]
    for flag, value in base_scalar_args.items():
        append_scalar_arg(cmd, flag, value)
    for flag, enabled in base_bool_flags.items():
        append_bool_flag(cmd, flag, enabled)
    for flag, enabled in optional_bool_flags.items():
        append_bool_flag(cmd, flag, enabled)
    if teacher_weights_path is not None:
        append_scalar_arg(cmd, '--supermix_weights', teacher_weights_path)
    if teacher_meta_path is not None:
        append_scalar_arg(cmd, '--supermix_meta', teacher_meta_path)

    if RUN_BENCHMARK_AFTER_TRAIN:
        append_scalar_arg(cmd, '--benchmark_eval_limit', BENCHMARK_EVAL_LIMIT)
    else:
        cmd.append('--skip_benchmark')

    latest_output_checkpoint = get_latest_adapter_checkpoint(OUTPUT_DIR)
    if latest_output_checkpoint is not None:
        print(f'Resuming from latest checkpoint in {OUTPUT_DIR}')
        cmd.append('--resume_from_latest_checkpoint')
    else:
        warm_start_checkpoint = seed_working_warm_start()
        if warm_start_checkpoint is not None:
            print(f'Warm-starting from checkpoint in {RESUME_WARM_START_DIR}')
            cmd.extend(['--init_adapter_dir', str(warm_start_checkpoint), '--init_adapter_match_lora'])
        else:
            print('No Kaggle warm-start artifact found. Starting a fresh LoRA run from the base model.')
    if EXTRA_ARGS:
        print('Applying extra args:', EXTRA_ARGS)
        cmd.extend(EXTRA_ARGS)
    return cmd


TRAIN_CMD = build_train_command()
OUT_LOG = LOG_DIR / f'train_{OUTPUT_DIR.name}.out.log'

launch_state = {
    'updated_at_utc': datetime.now(timezone.utc).isoformat(),
    'repo_url': REPO_URL,
    'branch': BRANCH,
    'workspace_dir': str(WORKSPACE_DIR),
    'output_dir': str(OUTPUT_DIR),
    'resume_warm_start_dir': str(RESUME_WARM_START_DIR),
    'source_dataset_slug': SOURCE_DATASET_SLUG,
    'training_dataset_slug': TRAINING_DATASET_SLUG,
    'teacher_dataset_slug': TEACHER_DATASET_SLUG,
    'warm_start_dataset_slug': WARM_START_DATASET_SLUG,
    'input_warm_start_dir': str(INPUT_WARM_START_DIR) if INPUT_WARM_START_DIR is not None else '',
    'base_model': BASE_MODEL,
    'device': DEVICE,
    'gpu_profile': gpu_profile,
    'detected_gpu_name': detected_gpu_name,
    'detected_gpu_memory_mb': detected_gpu_memory_mb,
    'out_log': str(OUT_LOG),
    'train_profile': TRAIN_PROFILE,
    'run_benchmark_after_train': RUN_BENCHMARK_AFTER_TRAIN,
    'teacher_weights_path': str(teacher_weights_path) if teacher_weights_path is not None else '',
    'teacher_meta_path': str(teacher_meta_path) if teacher_meta_path is not None else '',
    'command': TRAIN_CMD,
}
STATE_PATH.write_text(json.dumps(launch_state, indent=2), encoding='utf-8')
print('Command preview:')
print(' '.join(shlex.quote(part) for part in TRAIN_CMD))
print(json.dumps(launch_state, indent=2))


print('Streaming to:', OUT_LOG)
env = os.environ.copy()
env['PYTHONUNBUFFERED'] = '1'

with OUT_LOG.open('a', encoding='utf-8') as log_file:
    process = subprocess.Popen(
        TRAIN_CMD,
        cwd=str(WORKSPACE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end='')
        log_file.write(line)
        log_file.flush()
    return_code = process.wait()

if return_code != 0:
    raise RuntimeError(f'Training exited with code {return_code}')

print('Training finished successfully.')
