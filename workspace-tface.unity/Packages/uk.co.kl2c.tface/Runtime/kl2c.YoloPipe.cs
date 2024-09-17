
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

		public struct YoloFace
		{
			public Rect patch;
			public float detection;
			public float[] confidence;
		}

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
			string output = "outputs (" + runtimeModel.outputs.Count + ")";
			foreach (var o in runtimeModel.outputs)
				output += "\n\t-" + o;

			var shape = runtimeModel.GetShapeByName(runtimeModel.outputs[0]).Value.ToArray();
			Debug.Assert(8 == shape.Length);
			for (int i = 0; i < 6; ++i)
				Debug.Assert(1 == shape[i]);

			labelCount = shape[6] - 5;
			Debug.Assert(1 <= labelCount);
			Debug.Assert(2268 == shape[7]);

		}
		int labelCount;
		public IEnumerable<Rect> Execute(Texture inputTexture, float threshold, float[] confidence = null)
		{
			if (null != confidence)
				Debug.Assert(confidence.Length == labelCount);

			var classes = Enumerable.Range(0, labelCount).ToList();

			return Invoke(inputTexture)
				.Where(p =>
					(p.detection > threshold)
					&& (null == confidence || classes.All(i => confidence[i] >= p.confidence[i])))
					.Select(p =>
					{
						var patch = p.patch;
						//patch.y = inputTexture.height - patch.y;
						return patch;
					});
		}
		public IEnumerable<YoloFace> Invoke(Texture inputTexture)
		{
			Tensor inputTensor = new Tensor(inputTexture, channels: 3);

			// Execute the model with the input tensor
			worker.Execute(inputTensor);

			// Retrieve the output tensor
			Tensor outputTensor = worker.PeekOutput();

			// 
			foreach (var face in Transpose(5 + labelCount, outputTensor))
				yield return face;


			// Dispose of the input tensor to free resources
			inputTensor.Dispose();
			outputTensor.Dispose();
		}






		/// <summary>
		/// transpose the output tensor to the corrected form
		/// 
		/// this method is poorly named (maybe)
		/// </summary>
		/// <param name="width">5 + number of classes</param>
		/// <param name="floats">outputTensor.ToReadOnlyArray()</param>
		/// <returns></returns>
		private static IEnumerable<YoloFace> Transpose(int width, Tensor outputTensor)
		{
			var floats = outputTensor.ToReadOnlyArray();


			if (false) throw new Exception("??? https://github.com/FaceONNX/FaceONNX/blob/main/netstandard/FaceONNX/face/classes/FaceDetector.cs#L130-L233");

			var l = floats.Length;
			var count = l / width;

			Debug.Assert(
				// if this fails then the dimensionality of the output it wrong
				0 == (l % width));

			// pre-compute some indicies here
			var body = Enumerable.Range(0, width).Select(h => h * count);
			var head = body.Take(5).ToArray();
			var tail = body.Drop(5).ToArray();

			return Enumerable.Range(0, count).Fork(i =>
			{
				// compute the entry
				var entry = head
					.Select(h => floats[i + h])
					.ToArray();

				// create the result valeu thing
				var face = new YoloFace()
				{
					patch = new Rect()
					{
						x = entry[0],
						y = entry[1],
						width = (entry[2] - entry[0]),
						height = (entry[3] - entry[1]),
					},
					detection = entry[4],
					confidence = tail.Select(h => floats[i + h]).ToArray()
				};

				Debug.Assert(face.confidence.Length == (width - 5));
				return face;
			});
		}

		public void Dispose()
		{
			worker?.Dispose();
		}
	}

}
