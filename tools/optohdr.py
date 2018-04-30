# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Convert Intel ops to methods for assembler.

This tools converts the instruction op code table to a header file with methods
for encoding each instruction.

python tools/optohdr.py > third_party/jit/avx512.h

"""

import string

warnings = False

class Method:
  def __init__(self, name, opcode, args, numargs, flags):
    self.name = name
    self.opcode = opcode
    self.args = list(args)
    self.numargs = numargs
    self.flags = list(flags)
    self.mask = False
    self.bcst = False
    self.er = False
    self.sae = False
    self.ireg = -1

  def add_flag(self, flag):
    if flag not in self.flags: self.flags.append(flag)

  def add_flags(self, flags):
    for flag in flags: self.add_flag(flag)

  def clone(self):
    m = Method(self.name, self.opcode, self.args, self.numargs, self.flags)
    m.mask = self.mask
    m.bcst = self.bcst
    m.er = self.er
    m.sae = self.sae
    m.ireg = self.ireg
    return m

methods = []

def find_method(name, args):
  for m in methods:
    if m.name == name and m.args == args:
      return m
  return None

# Parse all the op definitions.
print "// Auto-generated from Intel instruction tables."
fin = open("third_party/jit/avx512ops.txt", "r")
for line in fin.readlines():
  line = line.strip()
  if len(line) == 0 or line[0] == "#": continue
  fields = line.replace(",", "").split("\t")
  enc = fields[0]
  tokens = enc.split(" ")

  mnemonic = ""
  opcode = ""
  flags = []
  imm8 = False
  vsib = False

  # Parse EVEX encoding.
  for flag in tokens[0].split("."):
    if flag == "EVEX":
      pass
    elif flag == "NDS" or flag == "NDD" or flag == "DDS":
      flags.append("EVEX_E" + flag);
    elif flag == "128" or flag == "256" or flag == "512":
      flags.append("EVEX_L" + flag);
    elif  flag == "LIG":
      flags.append("EVEX_LIG");
    elif flag == "66" or flag == "F2" or flag == "F3":
      flags.append("EVEX_P" + flag);
    elif flag == "0F" or flag == "0F38" or flag == "0F3A":
      flags.append("EVEX_M" + flag);
    elif flag == "W0" or flag == "W1" or flag == "WIG":
      flags.append("EVEX_" + flag);
    else:
      print "flag:", flag

  # Parse op code.
  opcode = tokens[1]

  # Parse rest of encoding
  i = 2
  ireg = -1
  while i < len(tokens):
    if tokens[i] == "/r":
      pass
    elif tokens[i] == "ib":
      imm8 = True
    elif tokens[i] in ["/0", "/1", "/2", "/3", "/4", "/5", "/6", "/7"]:
      ireg = int(tokens[i][1])
    elif tokens[i] == "/vsib":
      vsib = True
    elif all(c in string.hexdigits for c in tokens[i]):
      opcode = tokens[i] + opcode
    else:
      break
    i += 1

  # Get opcode mnemonic.
  mnemonic = tokens[i].lower()
  i += 1
  arguments = tokens[i:]

  # vsib encoding not supported.
  if vsib:
    if warnings: print "// vsib encoding not supported for " + mnemonic
    continue

  # Parse instruction arguments.
  args = []
  mask = False
  bcst = False
  er = False
  sae = False
  numargs = 0
  for a in arguments:
    arg = ""
    for c in a:
      arg += "0" if c.isdigit() else c

    if arg in ["{k0}{z}", "{k0}"]:
      mask = True
    elif arg in ["k0"]:
      args.append("opmask")
      numargs += 1
    elif arg in ["xmm0", "ymm0", "zmm0"]:
      args.append("zmm")
      numargs += 1
    elif arg in ["zmm0{sae}"]:
      args.append("zmm")
      sae = True
      numargs += 1
    elif arg in ["m00", "m000"]:
      args.append("mem")
      numargs += 1
    elif arg in ["xmm0/m000",
                 "ymm0/m000",
                 "zmm0/m000",
                 "xmm0/m00",
                 "xmm0/m000"]:
      args.append("zmm/mem")
      numargs += 1
    elif arg in ["xmm0/m000/m00bcst",
                 "xmm0/m00/m00bcst",
                 "ymm0/m000/m00bcst",
                 "zmm0/m000/m00bcst"]:
      args.append("zmm/mem")
      bcst = True
      numargs += 1
    elif arg in ["zmm0/m000/m00bcst{er}"]:
      args.append("zmm/mem")
      bcst = True
      er = True
      numargs += 1
    elif arg in ["xmm0/m00{er}"]:
      args.append("zmm/mem")
      er = True
      numargs += 1
    elif arg in ["xmm0/m00",
                 "xmm0/m000",
                 "ymm0/m000",
                 "zmm0/m000"]:
      args.append("zmm/mem")
      numargs += 1
    elif arg in ["xmm0/m00{sae}",
                 "xmm0/m00{sae}",
                 "ymm0/m000{sae}"]:
      args.append("zmm/mem")
      sae = True
      numargs += 1
    elif arg in ["zmm0/m000/m00bcst{sae}", "ymm0/m000/m00bcst{sae}"]:
      args.append("zmm/mem")
      bcast = True
      sae = True
      numargs += 1
    elif arg in ["imm0"]:
      args.append("imm")
    elif arg in ["r00", "r00", "reg"]:
      args.append("reg")
      numargs += 1
    elif arg in ["reg/m00", "r00/m00", "r00/m00", "r/m00"]:
      args.append("reg/mem")
      numargs += 1
    elif arg in ["r/m00{er}"]:
      args.append("reg/mem")
      er = True
      numargs += 1
    else:
      args.append("XXX " + arg)

  method = find_method(mnemonic, args)
  if method == None:
    method = Method(mnemonic, opcode, args, numargs, flags)
    methods.append(method)
  else:
    if warnings and opcode != method.opcode:
      print "// Hmm! opcode mismatch", method.name, method.opcode
    if warnings and numargs != method.numargs:
      print "// Hmm! numargs mismatch"
    method.add_flags(flags)

  if ireg != -1: method.ireg = ireg
  if mask: method.mask = True
  if bcst: method.bcst = True
  if er: method.er = True
  if sae: method.sae = True

fin.close()

# Split methods that take reg/mem arguments.
for method in methods:
  reg_mem_arg = -1
  for i in range(len(method.args)):
    arg = method.args[i]
    if arg == "zmm/mem" or arg == "reg/mem":
      if warnings and reg_mem_arg != -1: print "// Oops!! multi reg/mem"
      reg_mem_arg = i

  if reg_mem_arg != -1:
    mem_method = method.clone()
    dual = method.args[reg_mem_arg].split("/")
    method.args[reg_mem_arg] = dual[0]
    mem_method.args[reg_mem_arg] = dual[1]
    if bcst:
      bcst = False
      mem_method.bcst = True
    elif sae:
      mem_method.sae = True
    if not find_method(mem_method.name, mem_method.args):
      methods.append(mem_method)

# Generate instruction methods.
signatures = []
for method in sorted(methods, key=lambda x: x.name):
  argsigs = []
  argnames = ["dst", "src"] if method.numargs == 2 else ["dst", "src1", "src2"]
  masking = False
  imm = False
  for i in range(len(method.args)):
    arg = method.args[i]
    if arg == "opmask":
      argsigs.append("OpmaskRegister " + argnames[i]);
    elif arg == "zmm":
      argsigs.append("ZMMRegister " + argnames[i]);
    elif arg == "reg":
      argsigs.append("Register " + argnames[i]);
    elif arg == "mem":
      argsigs.append("const Operand &" + argnames[i]);
    elif arg == "imm":
      argsigs.append("int8_t imm8");
      imm = True
    else:
      argsigs.append("!" + arg);

  if method.mask:
    argsigs.append("Mask mask = nomask");
    masking = True

  if method.bcst:
    method.add_flag("EVEX_BCST");
  elif method.er:
    argsigs.append("RoundingMode er = kRoundToNearest");
    method.add_flag("EVEX_ER");
    method.add_flag("evex_round(er)");
  elif method.sae:
    method.add_flag("EVEX_SAE");

  body = "zinstr(0x" + method.opcode;
  if method.ireg != -1:
    body += ", zmm" + str(method.ireg)
  for i in range(method.numargs):
    body += ", " + argnames[i];
  if imm:
    body += ", imm8";
    method.add_flag("EVEX_IMM");
  else:
    body += ", 0";

  if masking:
    body += ", mask";
  else:
    body += ", nomask";
  body += ", " + " | ".join(sorted(method.flags))
  body += ");"

  sig = "void " + method.name + "(" + ", ".join(argsigs) + ")"
  if sig not in signatures:
    print sig + " {\n  " + body + "\n}"
    signatures.append(sig)

