
using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System.Linq;

namespace kl2c
{
    public class YoloPipe : IDisposable
    {
        public readonly Model runtimeModel;
        private IWorker worker;

        public (int, int) Size => (runtimeModel.inputs[0].shape[6], runtimeModel.inputs[0].shape[5]);

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

            Debug.Assert(3 == runtimeModel.inputs[0].shape[7]);

            // something(s) in here should dictate the "6" size, but, i dun't know what
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

            // 
            var tree =
                Transpose(6, outputTensor)
                    .Where(p => p.Item2 > detectionThreshold)
                    .Where(p => p.Item3[0] > confidenceThreshold)
                    .Select(p => p.Item1)
                    .ToList();

            // Dispose of the input tensor to free resources
            inputTensor.Dispose();
            outputTensor.Dispose();

            return tree;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="width">5 + number of classes</param>
        /// <param name="floats">outputTensor.ToReadOnlyArray()</param>
        /// <returns></returns>
        private static IEnumerable<(Rect, float, float[])> Transpose(int width, Tensor outputTensor)
        {
            var floats = outputTensor.ToReadOnlyArray();
            var l = floats.Length;
            var count = l / width;

            var body = Enumerable.Range(0, width).Select(h => h * count);

            var head = body.Take(5).ToArray();

            var tail = body.Drop(5).ToArray();

            return Enumerable.Range(0, count).Fork(i =>
            {
                // compute the entry
                var entry = head
                    .Select(h => floats[i + h])
                    .ToArray();

                // create teh result valeu thing
                return (new Rect()
                {
                    x = entry[0],
                    y = entry[1],
                    width = entry[2],
                    height = entry[3],
                }, entry[4], tail.Select(h => floats[i + h]).ToArray());
            });
        }

        public void Dispose()
        {
            worker?.Dispose();
        }
    }

}