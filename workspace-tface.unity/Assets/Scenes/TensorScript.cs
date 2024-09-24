
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System.IO;
using System.Drawing;
using System.Net.WebSockets;
using System.Linq;
using kl2c;
using UnityEngine.UI;
using UnityEngine.UIElements;

public class TensorScript : MonoBehaviour
{
	public NNModel modelAsset;

	public SpinCube cube1;
	public SpinCube cube2;
	void Start()
	{
		Debug.Assert(modelAsset != null, "Model asset not set in the Inspector");

		yoloPipe = new kl2c.YoloPipe(modelAsset);

		var (width, height) = yoloPipe.Size;

		inputTesorRenderTexture = new RenderTexture(width, height, 0);
	}

	public WebcamDisplay webcamDisplay;
	private RenderTexture inputTesorRenderTexture;
	public Texture2D outputTexture2D;


	[UnityEngine.Range(0, 1)]
	public float DetectionThreshold = 0.3f;

	[UnityEngine.Range(0, 1)]
	public float NmsThreshold = 0.5f;

	[UnityEngine.Range(0, 1)]
	public float ConfidenceThreshold = 0.4f;

	private kl2c.YoloPipe yoloPipe;


	public RawImage outputRawImage;

	void Update()
	{
		var webcamTexture = webcamDisplay.webcamTexture;
		// Check if the webcam has provided new data
		if (!webcamTexture.didUpdateThisFrame)
		{
			return;
		}

		// Convert the webcam texture to a Tensor
		Graphics.Blit(source: webcamTexture, dest: inputTesorRenderTexture);

		// do the deteaction, but, with NMS
		var detectionResults =
			yoloPipe.Invoke(inputTesorRenderTexture)

			// take out the results that're not confident enough
			.Where(p => p.detection > DetectionThreshold)


			.Select(p => p.patch)

			//
			.ToArray();

		// create thge output texture if we need to
		if (null == outputTexture2D)
			outputRawImage.texture = outputTexture2D = outputTexture2D = new Texture2D(inputTesorRenderTexture.width, inputTesorRenderTexture.height);

		// fill it with a solid colour
		outputTexture2D.Fill(UnityEngine.Color.black);

		// copy the webcamtex to the output

		int l = -1, t = -1; // l and t are used tro avoid drawing redundant pixels
		for (int i = 0; i < webcamTexture.width; i++)
		{
			int x = (int)((i / (float)webcamTexture.width) * outputTexture2D.width);
			if (x != l)
			{
				l = x;
				for (int j = 0; j < webcamTexture.height; j++)
				{
					int y = (int)((j / (float)webcamTexture.height) * outputTexture2D.height);

					if (y != t)
					{
						t = y;

						var q = i;
						var r = webcamTexture.height - j;

						var colour = webcamTexture.GetPixel(q, r);
						// if (0 <= x && x < outputTexture2D.width && 0 <= y && y < outputTexture2D.height)
							outputTexture2D.SetPixel(x, y, colour);
					}
				}
			}
		}




		//
		outputTexture2D.Confetti(detectionResults);
		// outputTexture2D.FaceOvals(detectionResults);

		// draw a yellow border to check my assumptions
		if (false)
			for (int i = 0; i < webcamTexture.width; i++)
				for (int j = 0; j < webcamTexture.height; j++)
					if (i < 5 || j < 5)
						outputTexture2D.SetPixel(i, j, UnityEngine.Color.yellow);//.GetPixel(i, j));




		// push the changes to the GPU
		outputTexture2D.Apply();
	}


	private void OnDestroy()
	{
		yoloPipe?.Dispose();
	}
}
