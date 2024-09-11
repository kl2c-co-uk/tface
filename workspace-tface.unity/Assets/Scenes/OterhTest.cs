using kl2c;
using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;

public class OterhTest : MonoBehaviour
{
	public Material otherImage;
	public Material otherResult;

	public NNModel modelAsset;
	[UnityEngine.Range(0, 1)]
	public float DetectionThreshold = 0.3f;
	[UnityEngine.Range(0, 1)]
	public float NmsThreshold = 0.5f;
	[UnityEngine.Range(0, 1)]
	public float ConfidenceThreshold = 0.4f;

	void Update()
	{
		if (!Input.GetButtonDown("Jump"))
			return;

		enabled = false;



		// build the pope thing
		using var yoloPipe = new kl2c.YoloPipe(modelAsset);
		var (width, height) = yoloPipe.Size;

		var inputRenderTexture = new RenderTexture(width, height, 0);

		// Convert the input texture to a Tensor
		Graphics.Blit(source: otherImage.mainTexture, dest: inputRenderTexture);

		// prepare the output putextr
		var outTexture = new Texture2D(inputRenderTexture.width, inputRenderTexture.height);
		otherResult.mainTexture = outTexture;

		var detectionResults = yoloPipe.Execute(inputRenderTexture, DetectionThreshold);

		// fill it randomly
		outTexture.Confetti(detectionResults);

		// push the changes to the GPU
		outTexture.Apply();

		Destroy(inputRenderTexture);
	}
}
