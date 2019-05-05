/*
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
//
// This header file defines EdgeTpuManager, and EdgeTpuContext.
// EdgeTpuContext is an object associated with one or more tflite::Interpreter.
// Instances of this class should be allocated through
// EdgeTpuManager::NewEdgeTpuContext.
// More than one Interpreter instances can point to the same context. This means
// the tasks from both would be executed under the same TPU context.
// The lifetime of this context must be longer than all associated
// tflite::Interpreter instances.
//
// Typical usage with NNAPI:
//
//   std::unique_ptr<tflite::Interpreter> interpreter;
//   tflite::ops::builtin::BuiltinOpResolver resolver;
//   auto model =
//   tflite::FlatBufferModel::BuildFromFile(model_file_name.c_str());
//
//   // Registers edge TPU custom op handler with Tflite resolver.
//   resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
//
//   tflite::InterpreterBuilder(*model, resolver)(&interpreter);
//
//   interpreter->AllocateTensors();
//      .... (Prepare input tensors)
//   interpreter->Invoke();
//      .... (retrieving the result from output tensors)
//
//   // Releases interpreter instance to free up resources associated with
//   // this custom op.
//   interpreter.reset();
//
// Typical usage with Non-NNAPI:
//
//   // Sets up the tpu_context.
//   auto tpu_context =
//       edgetpu::EdgeTpuManager::GetSingleton()->NewEdgeTpuContext();
//
//   std::unique_ptr<tflite::Interpreter> interpreter;
//   tflite::ops::builtin::BuiltinOpResolver resolver;
//   auto model =
//   tflite::FlatBufferModel::BuildFromFile(model_file_name.c_str());
//
//   // Registers edge TPU custom op handler with Tflite resolver.
//   resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
//
//   tflite::InterpreterBuilder(*model, resolver)(&interpreter);
//
//   // Binds a context with a specific interpreter.
//   interpreter->SetExternalContext(kTfLiteEdgeTpuContext,
//     tpu_context.get());
//
//   // Note that all edge TPU context set ups should be done before this
//   // function is called.
//   interpreter->AllocateTensors();
//      .... (Prepare input tensors)
//   interpreter->Invoke();
//      .... (retrieving the result from output tensors)
//
//   // Releases interpreter instance to free up resources associated with
//   // this custom op.
//   interpreter.reset();
//
//   // Closes the edge TPU.
//   tpu_context.reset();

#ifndef TFLITE_PUBLIC_EDGETPU_H_
#define TFLITE_PUBLIC_EDGETPU_H_

#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/context.h"

namespace edgetpu {

// EdgeTPU custom op.
static const char kCustomOp[] = "edgetpu-custom-op";

enum class DeviceType {
  kApexPci = 0,
  kApexUsb = 1,
};

// External context to be assigned through
// tflite::Interpreter::SetExternalContext.
class EdgeTpuContext : public TfLiteExternalContext {
 public:
  virtual ~EdgeTpuContext() = 0;
};

// Singleton edge TPU manager for allocating new TPU contexts.
class EdgeTpuManager {
 public:
  struct DeviceEnumerationRecord {
    DeviceType type;
    std::string path;
  };

  // Returns pointer to the singleton object, or nullptr if not supported on
  // this platform.
  static EdgeTpuManager* GetSingleton();

  // Creates a new Edge TPU context to be assigned to Tflite::Interpreter. The
  // Edge TPU context is associated with the default TPU device. May be null
  // if underlying device cannot be found or open. Caller owns the returned new
  // context and should destroy the context either implicity or explicitly after
  // all interpreters sharing this context are destroyed.
  virtual std::unique_ptr<EdgeTpuContext> NewEdgeTpuContext() = 0;

  // Same as above, but the created context is associated with the specified
  // type.
  virtual std::unique_ptr<EdgeTpuContext> NewEdgeTpuContext(
      DeviceType device_type) = 0;

  // Same as above, but the created context is associated with the specified
  // type and device path.
  virtual std::unique_ptr<EdgeTpuContext> NewEdgeTpuContext(
      DeviceType device_type, const std::string& device_path) = 0;

  // Same as above, but the created context is associated with the given device
  // type, path and options. Supported options are:
  //  - "Performance": ["Low", "Medium", "High", "Max"]
  virtual std::unique_ptr<EdgeTpuContext> NewEdgeTpuContext(
      DeviceType device_type, const std::string& device_path,
      const std::unordered_map<std::string, std::string>& options) = 0;

  // Enumerates all connected Edge TPU devices.
  virtual std::vector<DeviceEnumerationRecord> EnumerateEdgeTpu() const = 0;

  // Sets verbosity of operating logs related to edge TPU.
  // Verbosity level can be set to [0-10], in which 10 is the most verbose.
  virtual TfLiteStatus SetVerbosity(int verbosity) = 0;

  // Returns the version of EdgeTPU runtime stack.
  virtual std::string Version() const = 0;

 protected:
  // No deletion for this singleton instance.
  virtual ~EdgeTpuManager() = default;
};

// Returns pointer to an instance of TfLiteRegistration to handle
// EdgeTPU custom ops, to be used with
// tflite::ops::builtin::BuiltinOpResolver::AddCustom
TfLiteRegistration* RegisterCustomOp();

// Inserts name of device type into ostream. Returns the modified ostream.
std::ostream& operator<<(std::ostream& out, DeviceType device_type);

}  // namespace edgetpu

#endif  // TFLITE_PUBLIC_EDGETPU_H_
