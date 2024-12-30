// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

#include "module.h"
#include "func.h"

#include <memory>

namespace WasmEdge {
namespace Host {

WasiIOErrorModule::WasiIOErrorModule()
    : ComponentInstance("wasi:io/error@0.2.0") {
  addExport("error", IoError(getEnv()));
}

WasiIOStreamsModule::WasiIOStreamsModule()
    : ComponentInstance("wasi:io/streams@0.2.0") {
  addExport("input-stream", InputStream());
  addExport("output-stream", OutputStream());
  addHostFunc("[resource-drop]output-stream",
              std::make_unique<DropOutputStream>(Env));
  addExport("error", StreamError());
}

} // namespace Host
} // namespace WasmEdge
