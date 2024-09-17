
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine;
using UnityEngine.UI;

public class WebcamDisplay : MonoBehaviour
{
	public RawImage rawImage; // Reference to the UI RawImage component
	public WebCamTexture webcamTexture { private set; get; }

	public string deviceName = null;

	void Start()
	{
		// Initialize the webcam texture
		webcamTexture = new WebCamTexture();

		// list all and select selected if it's there
		foreach (var name in WebCamTexture.devices.Select(p => p.name))
		{
			Debug.Log("DeviceName =" + name + '=');
			if (name == deviceName)
				webcamTexture.deviceName = name;
		}


		// Assign the webcam texture to the RawImage component
		rawImage.texture = webcamTexture;

		// Start the webcam
		webcamTexture.Play();

		// save the name of whatever we connected to
		deviceName = webcamTexture.deviceName;

	}

	void OnDestroy()
	{
		// Stop the webcam when the object is destroyed
		if (webcamTexture != null && webcamTexture.isPlaying)
		{
			webcamTexture.Stop();
		}
	}
}
