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
using System.Threading.Tasks;
using Unity.VisualScripting;


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

            // peel out the data
            var floats = outputTensor.ToReadOnlyArray();
            var deku = FaceChopped.DekkuTree(outputTensor);
            var tree = FaceChopped.ReadTensor(
                detectionThreshold, nmsThreshold, confidenceThreshold,
                Size,
                // Assuming `output` is the tensor with shape (1, 1, 6, 25200)
                floats
                )
                    .Each(_ => _.Rectangle)
                    .Each(r => new Rect(r.X, r.Y, r.Width, r.Height))
                    .ToList();

            tree =
                Transpose(6, floats)
                    .Where(p => p.Item2 > detectionThreshold)
                    .Where(p => p.Item3[0] > confidenceThreshold)
                    .Select(p => p.Item1)
                    .ToList();

            // Dispose of the input tensor to free resources
            inputTensor.Dispose();
            outputTensor.Dispose();

            return tree;
        }

        private static IEnumerable<(Rect, float, float[])> Transpose(int width, Tensor outputTensor)
        {
            return Transpose(width, outputTensor.ToReadOnlyArray());
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="width">5 + number of classes</param>
        /// <param name="floats">outputTensor.ToReadOnlyArray()</param>
        /// <returns></returns>
        private static IEnumerable<(Rect, float, float[])> Transpose(int width, float[] floats)
        {
            Debug.Assert(6 == width); //
            int l = floats.Length;
            int count = l / width;

            var heads = Enumerable.Range(0, 5).Select(h => h * count).ToArray();

            var tail = Enumerable.Range(5, width).Select(h => h * count).ToArray();

            return Enumerable.Range(0, count).Fork(i =>
            {
                // compute the entry
                var entry = heads
                    .Select(h => floats[i + h])
                    .ToArray();

                var rect = new Rect();

                rect.x = entry[0];
                rect.y = entry[1];
                rect.width = entry[2];
                rect.height = entry[3];

                return (rect, entry[4], tail.Select(h => floats[i + h]).ToArray());
            });
        }

        public bool dump = false;

        public void Dispose()
        {
            worker?.Dispose();
        }
    }

}