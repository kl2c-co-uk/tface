
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine;
using UnityEngine.UI;

public class WebcamDisplay : MonoBehaviour
{
    public RawImage rawImage; // Reference to the UI RawImage component
    private WebCamTexture webcamTexture;

    void Start()
    {
        // Initialize the webcam texture
        webcamTexture = new WebCamTexture();

        // Assign the webcam texture to the RawImage component
        rawImage.texture = webcamTexture;

        // Start the webcam
        webcamTexture.Play();
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
