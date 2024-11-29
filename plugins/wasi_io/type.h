// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#pragma once

#include "base.h"

#include <memory>

namespace WasmEdge {
namespace Host {

class ToDebugString : public WasiIO<ToDebugString> {
public:
  ToDebugString(WasiIOEnvironment &HostEnv) : WasiIO(HostEnv) {}
  using IoError = uint32_t;
  Expect<void> body(IoError) { return {}; }
};

class IoError : public AST::Component::ResourceType {
public:
  IoError(WasiIOEnvironment &Env)
      : ResourceType(new Runtime::Instance::ComponentInstance("error")) {
    // to-debug-string: func() -> string;
    Impl->addHostFunc("to-debug-string", std::make_unique<ToDebugString>(Env));
  }
};

class InputStream : public AST::Component::ResourceType {
public:
  InputStream()
      : ResourceType(new Runtime::Instance::ComponentInstance("input-stream")) {
  }
};
class OutputStream : public AST::Component::ResourceType {
public:
  OutputStream()
      : ResourceType(
            new Runtime::Instance::ComponentInstance("output-stream")) {}
};

namespace StreamError {
// varian
using T = Component::Variant<uint32_t, Tuple<>>;

T lastOperationFailed(uint32_t IOErr) noexcept { return T(IOErr); }
T closed() noexcept { return T{}; }

AST::Component::VariantTy ast() noexcept {
  return AST::Component::VariantTy{
      AST::Component::Case("last-operation-failed",
                           AST::Component::PrimValType::U32),
      AST::Component::Case("closed")};
}

} // namespace StreamError

} // namespace Host
} // namespace WasmEdge
