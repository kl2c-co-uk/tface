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

            // build a transposed copy
            if (dump)
            {
                const int width = 6;

                var entries = floats.Length / width;

                Debug.Assert((entries * width) == floats.Length);

                var tops = Enumerable.Range(0, width).ToArray();



                for (int i = 0; i < width; ++i)
                {
                    var mine = tops.Each(t =>
                    {
                        return floats[i + (t * width)];
                    });


                    throw new UnityException(mine.Fold("i = " + i + ", and f = ")((l, r) => l + ", " + r));
                }
                throw new UnityException("???");
            }



            // write it row and col
            if (dump)
            {
                using var row = new StreamWriter("yolo-row.csv");
                using var col = new StreamWriter("yolo-col.csv");

                for (int i = 0; i < floats.Length;)
                {
                    var cell = floats[i] + ",";
                    row.Write(cell);
                    col.Write(cell);

                    i++;

                    if (0 == (i % 6))
                        row.Write("\n");
                    if (0 == (i % (floats.Length / 6)))
                        col.Write("\n");
                }
            }



            // Dispose of the input tensor to free resources
            inputTensor.Dispose();
            outputTensor.Dispose();

            return tree;
        }

        public bool dump = false;

        public void Dispose()
        {
            worker?.Dispose();
        }
    }

    public struct Patch
    {
        Rect rect;
        int label;
        float confidence;

    }

}