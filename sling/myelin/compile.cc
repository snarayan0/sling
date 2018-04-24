// Copyright 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "sling/base/init.h"
#include "sling/base/flags.h"
#include "sling/base/logging.h"
#include "sling/base/types.h"
#include "sling/file/file.h"
#include "sling/myelin/compute.h"
#include "sling/myelin/flow.h"
#include "sling/myelin/kernel/dragnn.h"
#include "sling/myelin/kernel/tensorflow.h"
#include "sling/string/ctype.h"
#include "sling/util/elf-writer.h"

DEFINE_string(flow, "", "Myelin flow file");
DEFINE_string(o, "", "ELF object output file for generated code");
DEFINE_string(hdr, "", "C++ header file for accessing model");
DEFINE_string(data, "", "Separate data file for storing parameters");
DEFINE_string(ns, "", "C++ name space for generated code");
DEFINE_bool(upper, true, "uppercase class names");
DEFINE_bool(dragnn, false, "register DRAGNN kernels");

DEFINE_bool(sse, true, "SSE support");
DEFINE_bool(sse2, true, "SSE2 support");
DEFINE_bool(sse3, true, "SSE3 support");
DEFINE_bool(sse41, true, "SSE 4.1 support");
DEFINE_bool(avx, false, "AVX support");
DEFINE_bool(avx2, false, "AVX2 support");
DEFINE_bool(fma3, false, "FMA3 support");

using namespace sling;
using namespace sling::myelin;

// Myelin linker for ahead-of-time compilation.
class AOTLinker : public Linker {
 public:
  struct Options {
    string flow_file;              // source flow file
    bool external_data = false;    // parameter data stored in external file
    bool uppercase_names = false;  // uppercase class names
    string ns;                     // C++ name space for generated code
  };

  AOTLinker(const Options &options);

  // Linker interface.
  void BeginCell(Cell *cell) override;
  void EndCell(Cell *cell,
               jit::CodeGenerator *generator,
               jit::Code *code,
               int data_size) override;
  void AddData(Tensor *data) override;

  // Add channel.
  void AddChannel(const string &name, Tensor *format);

  // Link sections.
  void Link();

  // Write object file.
  void Write(const string &filename);

  // Write header file.
  void WriteHeader(const string &filename);

  // Write data file.
  void WriteData(const string &filename);

 private:
  // Return sanitized name that is a legal C++ identifier.
  string sanitize(const string &name) {
    string sanitized;
    for (char c : name) {
      if (!ascii_isalnum(c)) c = '_';
      sanitized.push_back(c);
    }
    return sanitized;
  }

  // Return sanitized class name that is a legal C++ identifier.
  string sanitize_class_name(const string &name) {
    string sanitized;
    if (options_.uppercase_names) {
      bool upper = true;
      for (char c : name) {
        if (!ascii_isalnum(c)) {
          upper = true;
        } else {
          if (upper) {
            c = ascii_toupper(c);
            upper = false;
          }
          sanitized.push_back(c);
        }
      }
    } else {
      sanitized = sanitize(name);
    }
    return sanitized;
  }

  // Return mangled symbol for name.
  string mangle(const string &name, bool func) {
    string sanitized = sanitize(name);
    string buf = "_ZN";
    buf.append(std::to_string(options_.ns.size()));
    buf.append(options_.ns);
    buf.append(std::to_string(sanitized.size()));
    buf.append(sanitized);
    buf.append("E");
    if (func) buf.append("Pv");
    return buf;
  }

  // Linker options.
  Options options_;

  // ELF object file writer.
  Elf elf_;

  // Code section.
  Elf::Buffer code_{&elf_, ".text", ".rela.text",
                    SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR};

  // Read-only data section.
  Elf::Buffer rodata_{&elf_, ".rodata", nullptr, SHT_PROGBITS, SHF_ALLOC};

  // Uninitialized data section.
  Elf::Buffer bss_{&elf_, ".bss", nullptr, SHT_NOBITS, SHF_ALLOC | SHF_WRITE};

  // External symbols.
  std::unordered_map<string, Elf::Symbol *> symbols_;

  // Output stream for C++ header file.
  std::ostringstream header_;
};

AOTLinker::AOTLinker(const Options &options) : options_(options) {
  // Write prologue for header file.
  if (!options_.flow_file.empty()) {
    header_ << "// This file has been automatically generated from:\n"
            << "// " + options_.flow_file + "\n\n";
  }
  header_ << "#include <stdlib.h>\n"
          << "#include <stdint.h>\n"
          << "#include <string.h>\n"
          << "namespace " << options_.ns << " {\n\n";

  // Allocation functions.
  header_ << "inline char *memalloc(int size, int align) { "
          << "char *data; "
          << "return posix_memalign(reinterpret_cast<void **>(&data), "
          << "align, size) == 0 ? data : 0;}\n"
          << "inline void memfree(char *data) { free(data); }\n"
          << "inline char *memrealloc(char *data, size_t old_size, "
          << "size_t new_size, size_t align) {\n"
          << "  char *buffer = memalloc(new_size, align);\n"
          << "  if (data) { memcpy(buffer, data, old_size); memfree(data); }\n"
          << "  return buffer;\n}\n";
}

void AOTLinker::BeginCell(Cell *cell) {
  // Align code buffer before generating new cell computation function.
  code_.Align(16);

  // Entry point for cell function.
  string entry_name = sanitize(cell->name() + "_entry");
  header_ <<  "extern void " << entry_name << "(void *data);\n\n";

  // Write class prologue for cell.
  string cname = sanitize_class_name(cell->name());
  header_ << "// " << cell->name() << "\n"
          << "class " + cname + " {\n"
          << " public:\n";

  // Instance constructor.
  header_ << "  " << cname << "() { "
          << "data_ = memalloc("
          << cell->instance_size() << ", " << cell->instance_alignment()
          << "); }\n";

  // Instance destructor.
  header_ << "  ~" + cname + "() { memfree(data_); }\n";

  // Clear method.
  header_ << "  void clear() { memset(data_, 0, " << cell->instance_size()
          << "); }\n";

  // Compute method.
  header_ << "  void compute() { " << entry_name << "(data_); }\n";
  header_ << "\n";

  // Accessors for input/output tensors.
  for (Tensor *t : cell->network()->parameters()) {
    // Find input and output tensors for cell.
    if (t->cell() != cell) continue;
    if (!t->in() && !t->out()) continue;

    // Strip cell name from tensor name.
    string name = t->name();
    int len = cell->name().size();
    if (name.size() > len &&
        name.substr(0, len) == cell->name() &&
        name[len] == '/') {
      name = name.substr(len + 1);
    }
    name = sanitize(name);

    const char *ctype = TypeTraits::of(t->type()).ctype();
    header_ << "  // " << t->name() << " " << t->TypeString();
    if (t->in()) header_ << " in";
    if (t->out()) header_ << " out";
    header_ << "\n";
    if (t->ref()) {
      // Accessor for getting reference to tensor.
      header_ << "  " << ctype << " *&" << name << "() {";
      header_ << " return *reinterpret_cast<" << ctype << " **>(data_ + "
              << t->offset() << "); }\n";
    } else {
      // Accessor for getting element.
      header_ << "  " << ctype << " &" << name << "(";
      for (int d = 0; d < t->rank(); ++d) {
        if (d > 0) header_ << ", ";
        header_ << "int d" << d;
      }
      header_ << ") {";
      header_ << " return *reinterpret_cast<" << ctype << " *>(data_ + "
              << t->offset();
      for (int d = 0; d < t->rank(); ++d) {
        header_ << " + d" << d << " * " << t->stride(d);
      }
      header_ << "); }\n";
    }
    header_ << "\n";
  }
}

void AOTLinker::EndCell(Cell *cell,
                        jit::CodeGenerator *generator,
                        jit::Code *code,
                        int data_size) {
  // Add entry point for cell computation.
  int code_size = generator->size() - data_size;
  string entry_name = mangle(cell->name() + "_entry", true);
  elf_.AddSymbol(entry_name.c_str(), code_.progbits, STB_GLOBAL, STT_FUNC,
                 code_size, code_.offset());

  // Add symbol for constant data.
  if (data_size > 0) {
    string data_name = mangle(cell->name() + "_data", false);
    elf_.AddSymbol(data_name.c_str(), code_.progbits, STB_LOCAL, STT_OBJECT,
                   data_size, code_.offset() + code_size);
  }

  // Output code to code section.
  int code_start = code_.offset();
  code_.Add(generator->begin(), generator->size());

  // Add relocations for external references.
  for (auto &e : generator->externs()) {
    // Try to find existing symbol in object file. If symbol is not known,
    // a new undefined symbol is added.
    Elf::Symbol *sym = symbols_[e.symbol];
    if (sym == nullptr) {
      sym = elf_.AddSymbol(e.symbol.c_str(), nullptr,
                           STB_GLOBAL, STT_NOTYPE);
      symbols_[e.symbol] = sym;
    }

    // Add relocations to code.
    for (int offset : e.refs) {
      code_.AddReloc(sym, R_X86_64_64, 0, code_start + offset);
      code_.Clear64(code_start + offset);
    }
  }

  // Write class prologue for cell.
  header_ << " private:\n"
          << "  char *data_;\n"
          << "};\n\n";
}

void AOTLinker::AddData(Tensor *data) {
  // Construct identifier for tensor.
  string data_name = mangle(data->name(), false);

  // Determine section for data.
  auto *s = options_.external_data ? &bss_ : &rodata_;

  // Ensure alignment of tensor data.
  s->Align(data->byte_alignment());

  // Add symbol for data block.
  Elf::Symbol *sym = elf_.AddSymbol(data_name.c_str(), s->progbits,
                                    STB_LOCAL, STT_OBJECT,
                                    data->space(), s->offset());
  symbols_[data->name()] = sym;

  // Output tensor to data section.
  s->Add(data->data(), data->space());
}

void AOTLinker::AddChannel(const string &name, Tensor *format) {
  int align = jit::CPU::CacheLineSize();
  if (format->byte_alignment() > align) align = align;

  string cname = sanitize_class_name(name);
  const char *ctype = TypeTraits::of(format->type()).ctype();
  int size = format->size();
  header_ << "// " << name << " channel\n"
          << "class " + cname + " {\n"
          << " public:\n"
          << "  " << cname << "() : data_(0), size_(0), capacity_(0)  {}\n"
          << "  ~" << cname << "() { memfree(data_); }\n"
          << "  " << ctype << " *operator [](int idx) const { "
          << "return reinterpret_cast<" << ctype << " *>(data_ + idx * "
          << size << " ); }\n"
          << "  int size() const { return size_; }\n"
          << "  void reserve(int n) {\n"
          << "    if (n < size_ || n == capacity_) return;\n"
          << "    data_ = memrealloc(data_, size_ * " << size << ", "
          << "n * " << size << ", " << align << ");\n"
          << "    capacity_ = n;\n"
          << "  }\n"
          << "  void resize(int n) {\n"
          << "    if (n > capacity_) {\n"
          << "      int cap = capacity_ * 2;\n"
          << "      if (cap < 8) cap = 8;\n"
          << "      if (cap < n) cap = n;\n"
          << "      reserve(cap);\n"
          << "    }\n"
          << "    size_ = n;\n"
          << "  }\n"
          << " private:\n"
          << "  char *data_;\n"
          << "  int size_;\n"
          << "  int capacity_;\n"
          << "};\n\n";
}

void AOTLinker::Link() {
  // Write epilogue for header file.
  auto *s = options_.external_data ? &bss_ : &rodata_;
  int size = s->offset();
  header_ << "extern char data[" << size << "];\n\n"
          << "}  // namespace " + options_.ns + "\n";

  // Output symbol for model data.
  string data = mangle("data", false);
  elf_.AddSymbol(data.c_str(), s->progbits, STB_GLOBAL, STT_OBJECT, size, 0);

  // Update sections.
  code_.Update();
  rodata_.Update();
  bss_.Update();
  elf_.Update();
}

void AOTLinker::Write(const string &filename) {
  elf_.Write(filename.c_str());
}

void AOTLinker::WriteHeader(const string &filename) {
  CHECK(File::WriteContents(filename, header_.str()));
}

void AOTLinker::WriteData(const string &filename) {
  CHECK(File::WriteContents(filename, bss_.content));
}

static string basename(const string &name) {
  string base = name;
  int begin = base.rfind('/');
  if (begin != -1) base = base.substr(begin + 1);
  int end = base.find('.');
  if (end != -1) base = base.substr(0, end);
  return base;
}

void ConfigureCPUFeature(jit::CpuFeature feature, bool enabled) {
  if (enabled) {
    jit::CPU::Enable(feature);
  } else {
    jit::CPU::Disable(feature);
  }
}

int main(int argc, char *argv[]) {
  InitProgram(&argc, &argv);

  // Set up CPU features for compilation.
  ConfigureCPUFeature(jit::SSE, FLAGS_sse || FLAGS_sse2 || FLAGS_sse3);
  ConfigureCPUFeature(jit::SSE2, FLAGS_sse2 || FLAGS_sse3);
  ConfigureCPUFeature(jit::SSE3, FLAGS_sse3);
  ConfigureCPUFeature(jit::SSE4_1, FLAGS_sse41);
  ConfigureCPUFeature(jit::AVX, FLAGS_avx || FLAGS_avx2);
  ConfigureCPUFeature(jit::AVX2, FLAGS_avx2);
  ConfigureCPUFeature(jit::FMA3, FLAGS_fma3);

  // Set up kernel library.
  Library library;
  RegisterTensorflowLibrary(&library);
  if (FLAGS_dragnn) RegisterDragnnLibrary(&library);

  // Load flow.
  Flow flow;
  CHECK(flow.Load(FLAGS_flow));

  // Analyze flow.
  flow.Analyze(library);

  // Set up linker.
  AOTLinker::Options linker_opts;
  if (!FLAGS_ns.empty()) {
    linker_opts.ns = FLAGS_ns;
  } else {
    linker_opts.ns = basename(FLAGS_flow);
  }
  if (!FLAGS_data.empty()) linker_opts.external_data = true;
  linker_opts.uppercase_names = FLAGS_upper;
  linker_opts.flow_file = FLAGS_flow;
  AOTLinker linker(linker_opts);

  // Compile flow.
  Network net;
  net.set_linker(&linker);
  if (!net.Compile(flow, library)) {
    std::cout << "Compilation of flow failed\n";
    return 1;
  }

  // Add channels.
  for (Flow::Connector *cnx : flow.cnxs()) {
    if (cnx->links.empty()) continue;
    Tensor *format = net.GetParameter(cnx->links[0]->name);
    linker.AddChannel(cnx->name, format);
  }

  // Write ELF object file.
  if (!FLAGS_o.empty()) {
    linker.Link();
    linker.Write(FLAGS_o);
  }

  // Write header file.
  if (!FLAGS_hdr.empty()) {
    linker.WriteHeader(FLAGS_hdr);
  }

  // Write parameter data file.
  if (!FLAGS_data.empty()) {
    linker.WriteData(FLAGS_data);
  }

  return 0;
}

