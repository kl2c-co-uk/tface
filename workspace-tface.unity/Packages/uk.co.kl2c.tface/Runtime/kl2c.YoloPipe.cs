
using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System.Linq;
using System.IO;

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

		public IEnumerable<Rect> Execute(Texture inputTexture, float threshold, float[] confidenceThreshold = null)
		{
			if (null != confidenceThreshold)
				Debug.Assert(confidenceThreshold.Length == labelCount);
			Tensor inputTensor = new Tensor(inputTexture, channels: 3);

			// Execute the model with the input tensor
			worker.Execute(inputTensor);

			// Retrieve the output tensor
			Tensor outputTensor = worker.PeekOutput();

			// 
			var tree =
				Transpose(5 + labelCount, outputTensor)
					.Where(p => p.Item2 >= threshold)
					.Where(p =>
					{
						var confidenceValue = p.Item3;
						Debug.Assert(confidenceValue.Length == labelCount);
						return (Enumerable.Range(0, labelCount).Where(i => confidenceValue[i] >= (null == confidenceThreshold ? threshold : confidenceThreshold[i])).ToList().Count > 0);
					})
					.Select(p => p.Item1)
					.Select(patch =>
					{
						patch.y = inputTexture.height - patch.y;
						return patch;
					})
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
