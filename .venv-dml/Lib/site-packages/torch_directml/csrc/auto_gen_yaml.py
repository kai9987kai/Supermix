import yaml
import argparse

GLOBAL_OP_DICT = {}

def record_op(op, priority):
  op = op.replace('_', '\\_')
  if op in GLOBAL_OP_DICT:
    GLOBAL_OP_DICT[op] = min(GLOBAL_OP_DICT[op], priority)
  else:
    GLOBAL_OP_DICT[op] = priority

def save_op_analysis():
  import pandas as pd
  import os
  columns = ['op_name', 'supported_status']
  icons = ['✅', '🚧', '⛔', '❌', ]
  # sort the op dictionary based alphabetic order
  sorted_op_dict = dict(sorted(GLOBAL_OP_DICT.items(), key=lambda k: (k[1], k[0].lower().replace('\\_', ''))))
  for k, v in sorted_op_dict.items():
    sorted_op_dict[k] = icons[v]

  df = pd.DataFrame(sorted_op_dict.items(), columns=columns)
  parent_folder = os.path.dirname(os.path.abspath(__file__))
  save_des = os.path.join(parent_folder, "..", "..", "roadmap", "roadmap_from_yaml.md")
  with open(save_des, 'w',  encoding="utf-8") as f:
      f.write(df.to_markdown())

def get_dml_supported_ops(dml_yaml_path):
  # return supported ops in DML as a python list
  # ['abs', 'abs_', 'abs.out', '_softmax', ...]
  with open(dml_yaml_path) as f:
      dict = yaml.load(f, Loader=yaml.FullLoader)

  dml_supported_ops = dict["supported"]
  return dml_supported_ops

def parse_latest_torch_yaml(torch_yaml_path):
  with open(torch_yaml_path) as f:
      dict = yaml.load(f, Loader=yaml.FullLoader)

  def get_op_name_from_func(func_str):
    # "erfc(Tensor self) -> Tensor" -> "erfc"
    return func_str.split('(')[0]
  
  def get_all_dispatch_keys(dispatch_dict):
    # {'CPU, CUDA': 'all_out', 'MPS': 'all_out_mps'} -> ['CPU', 'CUDA', 'MPS']
    # {'CPU': 'add_relu_'}                           -> ['CPU']
    dispatch_keys = []
    for key in dispatch_dict:
      dispatch_keys.extend("".join(key.split()).split(','))
    return dispatch_keys
    
  all_aten_ops = [get_op_name_from_func(x['func']) for x in dict]

  '''
  - func: _softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor
    structured_delegate: _softmax_backward_data.out
  '''
  all_structured_delegate_ops = {get_op_name_from_func(x['func']):x['structured_delegate'] for x in dict if x.__contains__('structured_delegate')}

  '''
  - func: count_nonzero.dim_IntList(Tensor self, int[] dim) -> Tensor
    variants: function, method
    dispatch:
      CPU: count_nonzero_cpu
      CUDA: count_nonzero_cuda
      MPS: count_nonzero_mps
    autogen: count_nonzero.dim_IntList_out
  '''
  all_autogen_ops = {x['autogen']:get_op_name_from_func(x['func']) for x in dict if x.__contains__('autogen')}

  '''
  - func: _softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype, *, Tensor(a!) grad_input) -> Tensor(a!)
    structured: True
    dispatch:
      CPU: softmax_backward_cpu_out
      CUDA: softmax_backward_cuda_out
      MPS: softmax_backward_mps_out
  '''
  all_dispatch_keys = {get_op_name_from_func(x['func']):get_all_dispatch_keys(x['dispatch']) for x in dict if x.__contains__('dispatch')}

  all_aten_ops.extend(all_autogen_ops.keys())
  all_aten_ops = sorted(all_aten_ops, key = lambda kv: kv.lower().replace('_', '').replace('native', ''))

  return all_aten_ops, all_structured_delegate_ops, all_autogen_ops, all_dispatch_keys

def comment_unsupported_ops(save_des):
  # comment out unsupported operators in dml_native_functions.yaml
  fp_r = open(save_des, 'r')
  lines = fp_r.readlines()
  for idx in range(len(lines)):
    if '#' in lines[idx]:
      lines[idx] = lines[idx].replace('#', '    #').replace('-', '## -')
  fp_r.close()
  fp_w = open(save_des, 'w')
  fp_w.write(''.join(lines))
  fp_w.close()

def tag_op(op, all_structured_delegate_ops, all_autogen_ops, all_dispatch_keys, dml_supported_ops):
  # tag operators with instructive info
  '''
  - absolute.out    #(no_real_kernel)
  - adaptive_avg_pool3d.out    #(CPU_CUDA_dispatch)
  - adaptive_max_pool2d    #(adaptive_max_pool2d.out)
  - addr_    #(default_backend)
  - to_sparse_bsr.out #(autogen)
  '''
  op_with_tag = op
  visited = False
  
  # if op is delegated, check if the delegated op has DML registration
  if op in all_structured_delegate_ops:
    '''
      - func: sgn(Tensor self) -> Tensor
        variants: function, method
        structured_delegate: sgn.out
        dispatch:
          SparseCPU, SparseCUDA: sgn_sparse
          SparseCsrCPU, SparseCsrCUDA: sgn_sparse_csr
    '''
    visited = True
    op_with_tag += "#(delegated to %s)"% all_structured_delegate_ops[op]
    if op in dml_supported_ops:
      record_op(op, 0)
      if all_structured_delegate_ops[op] in dml_supported_ops:
        record_op(all_structured_delegate_ops[op], 0)
        print('WARINING: Unnecesary registration, [%s] is delegated to [%s]' %(op, all_structured_delegate_ops[op]))
      else:
        print('WARINING: Wrong registration,  [%s] is delegated to [%s]' %(op, all_structured_delegate_ops[op]))
    else:
      if all_structured_delegate_ops[op] in dml_supported_ops:
        record_op(all_structured_delegate_ops[op], 0)
        record_op(op, 0)
      else:
        record_op(all_structured_delegate_ops[op], 1)
        record_op(op, 1)
        

  # if op has dispatch keys
  if op in all_dispatch_keys:
    '''
      - func: sgn.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
        structured: True
        structured_inherits: TensorIteratorBase
        dispatch:
          CPU, CUDA: sgn_out
          SparseCPU, SparseCUDA: sgn_sparse_out
          SparseCsrCPU, SparseCsrCUDA: sgn_sparse_csr_out
    '''
    visited = True

    CPU_CUDA_dispatch_or_default_backend = False
    if 'CPU' in all_dispatch_keys[op] or 'CUDA' in all_dispatch_keys[op]:
      CPU_CUDA_dispatch_or_default_backend = True
      if 'CPU' in all_dispatch_keys[op] and 'CUDA' in all_dispatch_keys[op]:
        op_with_tag += "#(CPU_CUDA_dispatch)"
        record_op(op, 1)
      elif 'CPU' in all_dispatch_keys[op]:
        op_with_tag += "#(CPU_only_dispatch)"
        record_op(op, 1)
      else:
        op_with_tag += "#(CUDA_ony_dispatch)"
        record_op(op, 3)
      counted = True

    if 'CompositeExplicitAutograd' in all_dispatch_keys[op] or 'CompositeExplicitAutogradNonFunctional' in all_dispatch_keys[op] or 'CompositeImplicitAutograd' in all_dispatch_keys[op]:
      CPU_CUDA_dispatch_or_default_backend = True
      if op in dml_supported_ops:
        print('WARINING: default backend exists for [%s]' %(op))
      op_with_tag += "#(default_backend)"
      record_op(op, 2)

    if not CPU_CUDA_dispatch_or_default_backend:
      op_with_tag += "#(other_backends)"
      record_op(op, 3)
  
  if op in all_autogen_ops:
    '''
    - func: count_nonzero.dim_IntList(Tensor self, int[] dim) -> Tensor
      variants: function, method
      dispatch:
        CPU: count_nonzero_cpu
        CUDA: count_nonzero_cuda
        MPS: count_nonzero_mps
      autogen: count_nonzero.dim_IntList_out
    '''
    visited = True
    op_with_tag += "#(autogen)"
    record_op(op, 3)

  if not visited:
    '''
      - func: output_nr(Tensor self) -> int
        manual_cpp_binding: True
        variants: method

      - func: _version(Tensor self) -> int
        manual_cpp_binding: True
        variants: method

      - func: requires_grad_(Tensor(a!) self, bool requires_grad=True) -> Tensor(a!)
        manual_cpp_binding: True
        variants: method
    '''
    op_with_tag += "#(no_real_kernel)"
    record_op(op, 3)

  return op_with_tag

def generate_yaml(dml_yaml_path, torch_yaml_path):
  yaml_dict = {
                'backend': 'PrivateUse1',
                'cpp_namespace': 'torch_dml',
                'supported': []
              }

  dml_supported_ops = get_dml_supported_ops(dml_yaml_path)
  all_aten_ops, all_structured_delegate_ops, all_autogen_ops, all_dispatch_keys = parse_latest_torch_yaml(torch_yaml_path)

  for op in all_aten_ops:
    op_with_tag = tag_op(op, all_structured_delegate_ops, all_autogen_ops, all_dispatch_keys, dml_supported_ops)
  
    if op in dml_supported_ops:
      record_op(op, 0)
      yaml_dict['supported'].append(op)
    else:
      yaml_dict['supported'].append(op_with_tag)

  save_op_analysis()
  with open(dml_yaml_path, "w", encoding = "utf-8") as yaml_file:
    dump = yaml.dump(yaml_dict, default_flow_style = False, allow_unicode = True, encoding = None, sort_keys=False)
    yaml_file.write(dump)
  
  comment_unsupported_ops(dml_yaml_path)
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Generate backend stub files")
  parser.add_argument(
      "-t",
      "--torch_yaml",
      help="path to torch native_function yaml file containing operator external definitions",
  )
  parser.add_argument(
      "-d",
      "--dml_yaml",
      help="path to dml_native_function yaml file",
  )

  options = parser.parse_args()
  generate_yaml(options.dml_yaml, options.torch_yaml)
