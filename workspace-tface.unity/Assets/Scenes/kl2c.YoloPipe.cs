using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using UnityEngine;
using UnityEngine;
using Unity.Barracuda;
using System.Collections.Generic;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System.IO;
using System.Drawing;
using System.Net.WebSockets;
using System.Linq;


namespace kl2c
{
    public class YoloPipe : IDisposable
    {
        private Model runtimeModel;
        private IWorker worker;

        public YoloPipe(NNModel modelAsset)
        {
            // Create a runtime model
            runtimeModel = ModelLoader.Load(modelAsset);
            worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, runtimeModel);


            // check the model size
            Debug.Assert(1 == runtimeModel.inputs.Count);
            Debug.Assert(runtimeModel.inputs[0].shape.Length == 8);

            // these should be 1
            for (int i = 0; i < 5; ++i)
                Debug.Assert(1 == runtimeModel.inputs[0].shape[i]);

            var height = runtimeModel.inputs[0].shape[5];
            var width = runtimeModel.inputs[0].shape[6];

            Debug.Assert(3 == runtimeModel.inputs[0].shape[7]);
        }

        public IEnumerable<Rect> Execute(
            Texture inputTexture,
            float detectionThreshold,
            float nmsThreshold,
            float confidenceThreshold)
        {
            Tensor inputTensor = new Tensor(inputTexture, channels: 3);

            // Execute the model with the input tensor
            worker.Execute(inputTensor);

            // Retrieve the output tensor
            Tensor outputTensor = worker.PeekOutput();

            Debug.Log("hey! apply nms et al");

            // peel out the data
            var tree = FaceChopped.DekkuTree(outputTensor);

            // Dispose of the input tensor to free resources
            inputTensor.Dispose();
            outputTensor.Dispose();

            return tree;
        }

        public void Dispose()
        {
            worker?.Dispose();
        }
    }
}