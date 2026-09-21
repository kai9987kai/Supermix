#!/usr/bin/env bash
# v91 male-CNS connectome experiment: three matched warm starts from v89, run
# serially (this box cannot run two). Identical data, seed, batch order, LR
# curve and step budget; the arms differ only in the grafted branch.
#
#   A  v91_cns_connectome  ConnectomeCore wired like the male CNS (512 modules)
#   C  v91_control         no graft: plain continued training of v89
#   B  v91_cns_rewired     same core, degree-preserving rewired wiring (null)
#
# Order A, C, B: if the chain is cut short, A-vs-C (does the graft help at
# all) is the comparison worth having first.
set -u
cd "$(dirname "$0")/../.."

COMMON=(
  --steps 2500 --lr 0.00015 --new_param_lr_mult 20
  --init_from output/v89_corpus/v89_corpus.pt
  --corpus_jsonl datasets/v89/v89_combined.jsonl --min_response_characters 1
  --digit_tokens --sequence_length 128 --max_vocab 16384
  --hidden_size 320 --n_layers 5 --n_heads 8 --n_kv_heads 2 --n_routed_experts 64
  --turn_aligned_packing --checkpoint_every_improvement
  --eval_every 625 --accuracy_every 2500 --accuracy_problems 100
  --probe_max_new_tokens 112 --select_on accuracy --torch_threads 8
)
CNS=(--cns_core --cns_graph datasets/v91_malecns/malecns_modules_512.npz)

run_arm() {
  local name="$1"; shift
  mkdir -p "output/$name"
  echo "=== $(date -Is) start $name" >> output/v91_malecns/chain.log
  python source/train_supervised.py --max_restarts 4 -- \
    --run_name "$name" --output_dir "output/$name" "${COMMON[@]}" "$@" \
    > "output/$name/train.log" 2>&1
  echo "=== $(date -Is) end $name exit $?" >> output/v91_malecns/chain.log
}

run_arm v91_cns_connectome "${CNS[@]}" --cns_wiring connectome
run_arm v91_control
run_arm v91_cns_rewired "${CNS[@]}" --cns_wiring rewired
echo "=== $(date -Is) chain done" >> output/v91_malecns/chain.log
