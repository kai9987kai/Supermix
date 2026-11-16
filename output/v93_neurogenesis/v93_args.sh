#!/usr/bin/env bash
# The v93 command, shared by the preflight and the real run so the two cannot
# drift. Source this file; it defines V93_MODEL (architecture + graft + growth
# quotas) and V93_DATA (corpus, tokenizer, packing). Step budget, schedule and
# split fractions are given per run (see run_v93.sh and the preflight log).
#
# Architecture: v89's h320 / 5 layers / 64 experts top-2 / thinking core, plus
#   * one appended dense identity block (--grow_layers 1 -> 6 layers, layout pinned)
#   * 8 spare expert slots per MoE layer (born dead; neurogenesis fills them)
#   * the two-hemisphere male-CNS core: 384 modules per side (768) + 64 spare
#     slots, temporal (state carried along the sequence, 2 inner updates per
#     token), reading blocks 0-2, writing after blocks 2-4 and into the
#     thinking core, every gate at zero at step 0. (The appended block 5 is
#     not a write site: the config is validated before --grow_layers appends
#     it -- the first preflight failed on exactly that -- and a write after
#     block 4 reaches the same residual stream through an identity block.)
# Growth every 1,000 steps (each eval): 6 module splits, 32 synapses (>= 16
# commissural), 6 afferent + 6 efferent taps, expert births where one expert
# carries > 2x its fair share; apoptosis after two consecutive weak readings.
V93_MODEL=(
  --hidden_size 320 --n_layers 5 --n_heads 8 --n_kv_heads 2 --n_routed_experts 64
  --moe_spare_experts 8 --grow_layers 1
  --cns_core --cns_graph datasets/v93_malecns/malecns_hemispheres_384.npz --cns_wiring connectome
  --cns_nodes 832 --cns_spare_nodes 64 --cns_steps 2 --cns_temporal --cns_after_layer 2
  --cns_read_layers 0 1 2 --cns_write_layers 2 3 4 --cns_to_thinking
  --grow_modules 6 --grow_edges 32 --grow_taps 6 --grow_experts --prune_threshold 0.0001
  --witness_rows 8
)
V93_DATA=(
  --init_from output/v89_corpus/v89_corpus.pt --extend_vocab --max_new_tokens_vocab 4000
  --corpus_jsonl datasets/v93/v93_train_mix.jsonl --min_response_characters 1
  --digit_tokens --sequence_length 128 --max_vocab 16384
  --turn_aligned_packing --checkpoint_every_improvement
  --probe_max_new_tokens 112 --select_on accuracy --accuracy_problems 205 --torch_threads 8
)
