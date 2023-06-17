#!/usr/bin/env swift

import Foundation
import CoreML

print("hello, world!")

let modelPath  = CommandLine.arguments[1]
let optimizeForDevice = { 
   switch CommandLine.arguments[2] {
     case "ALL":
        return MLComputeUnits.all
     case "CPU_AND_NE":
        return MLComputeUnits.cpuAndNeuralEngine
      case "CPU_AND_GPU":
         return MLComputeUnits.cpuAndGPU
      case "CPU_ONLY":
         return MLComputeUnits.cpuOnly
      default:
         print("Parameter 2 should be either 'ALL', 'CPU_AND_NE', 'CPU_AND_GPU' or 'CPU_ONLY'")    
         exit (9)    
}
}


let savePath  = CommandLine.arguments[1].replacingOccurrences(of: "mlpackage", with: "mlmodelc")
print(modelPath)

let modelURL = URL(fileURLWithPath: modelPath, isDirectory: true);
let saveURL = URL(fileURLWithPath: savePath, isDirectory: true);
let compiledURL = try MLModel.compileModel(at: modelURL);
print (compiledURL)


let modelConfig = MLModelConfiguration()
modelConfig.computeUnits = optimizeForDevice();


print(savePath)
let fileManager = FileManager.default
_ = try fileManager.replaceItemAt(saveURL,
                                  withItemAt: compiledURL)

let model = try MLModel(contentsOf:saveURL, configuration:modelConfig)


