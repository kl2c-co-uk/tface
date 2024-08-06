using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using UnityEngine;
using UnityEngine;
using Unity.Barracuda;
using System.Collections.Generic;
using System.Linq;
public static class FaceChopped
{

	public static List<Rect> DekkuTree(Tensor outTensor)
	{
		List<Rect> boxes = new List<Rect>();
		List<int> indices = new List<int>();
		List<float> scores = new List<float>();

		int numRows = outTensor.shape[0]; // number of rows
		int numCols = outTensor.shape[1]; // number of columns

		Debug.Log(
			"numCols = " + numCols
			+ "\n\tnumRows = " + numRows  );
		for (int r = 0; r < numRows; r++)
		{
			float cx = outTensor[r, 0];
			float cy = outTensor[r, 1];
			float w = outTensor[r, 2];
			float h = outTensor[r, 3];
			float sc = outTensor[r, 4]; // sc = score? is this the likelyhood that SOMETHING is there?

			// Retrieve the confidence scores for the classes
			float maxV = float.NegativeInfinity;
			int maxIndex = -1;
			for (int i = 5; i < numCols; i++)
			{
				float conf = outTensor[r, i] * sc;
				if (conf > maxV)
				{
					maxV = conf;
					maxIndex = i;
				}
			}

			scores.Add(maxV);
			boxes.Add(new Rect(cx - w / 2, cy - h / 2, w, h));
			indices.Add(r);
		}


		Debug.Log("TODO - DetectionThreshold / NmsThreshold / ConfidenceThreshold stuff");

		// Use the boxes, scores, and indices as needed
		return boxes;
	}




	public static List<FaceONNX.FaceDetectionResult> ReadTensor(
		float DetectionThreshold,
		float NmsThreshold,
		float ConfidenceThreshold,

		(int, int) inputSize,

		float[] data)
	{

		using StreamWriter writer = new StreamWriter("face_onnx_way.csv");

		// https://github.com/FaceONNX/FaceONNX/blob/main/netstandard/FaceONNX/face/classes/FaceDetector.cs#L108


		var Labels = new string[] { "face" };
		var size = new Size(inputSize.Item1, inputSize.Item2);
		var width = size.Width; //  image[0].GetLength(1);
		var height = size.Height; //  image[0].GetLength(0);

		// yolo params
		var yoloSquare = 15;
		var classes = Labels.Length;
		var count = classes + yoloSquare;

		// post-processing
		var vector = data; // results[0].AsTensor<float>().ToArray();
		var length = vector.Length / count;
		var predictions = new float[length][];

		for (int i = 0; i < length; i++)
		{
			var prediction = new float[count];

			for (int j = 0; j < count; j++)
				prediction[j] = vector[i * count + j];

			predictions[i] = prediction;
		}

		var list = new List<float[]>();

		// seivining results
		for (int i = 0; i < length; i++)
		{
			var prediction = predictions[i];

			if (prediction[4] > DetectionThreshold)
			{
				var a = prediction[0];
				var b = prediction[1];
				var c = prediction[2];
				var d = prediction[3];

				prediction[0] = a - c / 2;
				prediction[1] = b - d / 2;
				prediction[2] = a + c / 2;
				prediction[3] = b + d / 2;

				//for (int j = yoloSquare; j < prediction.Length; j++)
				//{
				//	prediction[j] *= prediction[4];
				//}

				list.Add(prediction);
			}
		}

		// non-max suppression
		list = FaceONNX.NonMaxSuppressionExensions.AgnosticNMSFiltration(list, NmsThreshold);

		// perform
		predictions = list.ToArray();
		length = predictions.Length;

		// backward transform
		var k0 = (float)size.Width / width;
		var k1 = (float)size.Height / height;
		float gain = Mathf.Min(k0, k1);
		float p0 = (size.Width - width * gain) / 2;
		float p1 = (size.Height - height * gain) / 2;

		// collect results
		var detectionResults = new List<FaceONNX.FaceDetectionResult>();

		for (int i = 0; i < length; i++)
		{
			var prediction = predictions[i];
			var labels = new float[classes];

			for (int j = 0; j < classes; j++)
			{
				labels[j] = prediction[j + yoloSquare];
			}

			var max = UMapx.Core.Matrice.Max(labels, out int argmax);

			if (max > ConfidenceThreshold)
			{
				var rectangle = Rectangle.FromLTRB(
					(int)((prediction[0] - p0) / gain),
					(int)((prediction[1] - p1) / gain),
					(int)((prediction[2] - p0) / gain),
					(int)((prediction[3] - p1) / gain));

				var points = new Point[5];

				for (int j = 0; j < 5; j++)
				{
					points[j] = new Point
					{
						X = (int)((prediction[5 + 2 * j + 0] - p0) / gain),
						Y = (int)((prediction[5 + 2 * j + 1] - p1) / gain)
					};
				}

				var landmarks = new FaceONNX.Face5Landmarks(points);

				//writer.Write($"{rectangle.Left}, {rectangle.Top}, {rectangle.Right}, {rectangle.Top},\n");

				detectionResults.Add(new FaceONNX.FaceDetectionResult
				{
					Rectangle = rectangle,
					Id = argmax,
					Score = max,
					Points = landmarks
				});
			}
		}

		return detectionResults;

	}
}