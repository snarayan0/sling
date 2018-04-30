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

print "// Auto-generated from Intel instruction tables."
fin = open("avx512ops.txt", "r")
for line in fin.readlines():
  line = line.strip()
  if len(line) == 0 or line[0] == "#": continue
  fields = line.replace(",", "").split("\t")
  enc = fields[0]
  tokens = enc.split(" ")
  #print tokens

  mnemonic = ""
  opcode = ""
  flags = []
  imm8 = False
  numbers = [False] * 10
  vsib = False

  # Parse EVEX encoding
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

  # Parse op code
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
    elif tokens[i] == "02":
      opcode = tokens[i] + opcode
    elif tokens[i] == "/vsib":
      vsib = True
    else:
      break
    i += 1


  # vsib encoding not supported
  if vsib: continue

  mnemonic = tokens[i].lower()
  i += 1
  arguments = tokens[i:]

  if mnemonic in ["vcvttss2si", "vcvttsd2si", "vcvtss2usi", "vcvttss2usi", "vcvtsd2usi", "vcvtss2si", "vcvttsd2usi"]: continue

  #if len(opcode) > 2: print "// extended opcode for " + mnemonic + " " + opcode
  #if ireg != -1: print "// " + mnemonic + " has ireg encoding: ", ireg

  args = []
  mask = False
  bcst = False
  er = False
  sae = False
  numargs = 0
  for arg in arguments:
    if arg in ["{k1}{z}", "{k2}", "{k1}"]:
      mask = True
    elif arg in ["k1", "k2"]:
      args.append("opmask")
      numargs += 1
    elif arg in ["xmm1", "ymm1", "zmm1", "xmm2", "ymm2", "zmm2", "xmm0", "xmm3"]:
      args.append("zmm")
      numargs += 1
    elif arg in ["zmm2{sae}"]:
      args.append("zmm")
      sae = True
      numargs += 1
    elif arg in ["m64", "m128", "m256", "m512", "m32", "/m128"]:
      args.append("mem")
      numargs += 1
    elif arg in ["xmm3/m128", "ymm3/m256", "zmm3/m512", "xmm3/m32", "xmm2/m16", "xmm2/m32", "xmm2/m64", "xmm3/m64", "xmm2/mm128", "xmm1/m16"]:
      args.append("zmm/mem")
      numargs += 1
    elif arg in ["xmm3/m128/m32bcst", "ymm3/m256/m32bcst", "zmm3/m512/m32bcst", "xmm3/m128/m64bcst", "ymm3/m256/m64bcst", "zmm3/m512/m64bcst", "xmm2/m128/m32bcst", "ymm2/m256/m32bcst", "xmm2/m64/m32bcst", "xmm2/m128/m64bcst", "ymm2/m256/m64bcst", "zmm2/m512/m32bcst", "zmm2/m512/m64bcst", "xmm2xmm3/m128/m64bcst"]:
      args.append("zmm/mem")
      bcst = True
      numargs += 1
    elif arg in ["zmm3/m512/m64bcst{er}", "zmm3/m512/m32bcst{er}", "zmm2/m512/m32bcst{er}", "zmm2/m512/m64bcst{er}"]:
      args.append("zmm/mem")
      bcst = True
      er = True
      numargs += 1
    elif arg in ["xmm3/m64{er}", "xmm3/m32{er}", "xmm1/m32{er}", "xmm1/m64{er}"]:
      args.append("zmm/mem")
      er = True
      numargs += 1
    elif arg in ["xmm1/m32", "xmm1/m64", "xmm1/m128", "xmm2/m128", "ymm2/m256", "zmm2/m512", "ymm1/m256", "zmm1/m512", "xmm2/m8"]:
      args.append("zmm/mem")
      numargs += 1
    elif arg in ["xmm3/m32{sae}", "xmm3/m64{sae}", "xmm1/m32{sae}", "xmm2/m32{sae}", "xmm2/m64{sae}", "xmm1/m64{sae}", "ymm2/m256{sae}"]:
      args.append("zmm/mem")
      sae = True
      numargs += 1
    elif arg in ["zmm2/m512/m32bcst{sae}", "zmm3/m512/m64bcst{sae}", "zmm2/m512/m64bcst{sae}", "zmm3/m512/m32bcst{sae}", "ymm2/m256/m32bcst{sae}"]:
      args.append("zmm/mem")
      bcast = True
      sae = True
      numargs += 1
    elif arg in ["imm8"]:
      args.append("imm")
    elif arg in ["r32", "r64", "reg"]:
      args.append("reg")
      numargs += 1
    elif arg in ["reg/m32", "r32/m32", "r64/m64", "r/m32", "r/m32"]:
      args.append("reg/mem")
      numargs += 1
    elif arg in ["r/m32{er}", "r/m64{er}"]:
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
    #if opcode != method.opcode: print "Hmm! opcode mismatch", method.name, method.opcode, method.args, opcode, args
    if numargs != method.numargs: print "Hmm! numargs mismatch"
    method.add_flags(flags)

  if ireg != -1: method.ireg = ireg
  if mask: method.mask = True
  if bcst: method.bcst = True
  if er: method.er = True
  if sae: method.sae = True

for method in methods:
  reg_mem_arg = -1
  for i in range(len(method.args)):
    arg = method.args[i]
    if arg == "zmm/mem" or arg == "reg/mem":
      if reg_mem_arg != -1: print "Oops!!"
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

fin.close()

