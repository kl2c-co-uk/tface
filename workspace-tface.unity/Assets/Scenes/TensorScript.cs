
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System.IO;
using System.Drawing;
using System.Net.WebSockets;
using System.Linq;
using kl2c;

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

	public Material outputMaterial;
	private kl2c.YoloPipe yoloPipe;

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

		var detectionResults = yoloPipe.Execute(inputTesorRenderTexture, DetectionThreshold, NmsThreshold, ConfidenceThreshold);

		// create thge output texture if we need to
		if (null == outputTexture2D)
			outputMaterial.mainTexture = outputTexture2D = outputTexture2D = new Texture2D(inputTesorRenderTexture.width, inputTesorRenderTexture.height);

		// fill it randomly
		outputTexture2D.Fill(UnityEngine.Color.black);
		//outputTexture2D.Confetti(detectionResults);
		outputTexture2D.FaceOvals(detectionResults);

		// push the changes to the GPU
		outputTexture2D.Apply();
	}


	private void OnDestroy()
	{
		yoloPipe?.Dispose();
	}
}
