
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System.IO;
using System.Drawing;
using System.Net.WebSockets;
using System.Linq;

public class TensorScript : MonoBehaviour
{
    public NNModel modelAsset;
    private Model runtimeModel;
    private IWorker worker;

    public SpinCube cube1;
    public SpinCube cube2;
    void Start()
    {
        if (modelAsset == null)
        {
            Debug.LogError("Model asset not set in the Inspector");
            return;
        }

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


        //
        //
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

    void Update()
    {
        var webcamTexture = webcamDisplay.webcamTexture;
        // Check if the webcam has provided new data
        if (webcamTexture.didUpdateThisFrame)
        {
            // Convert the webcam texture to a Tensor
            Graphics.Blit(source: webcamTexture, dest: inputTesorRenderTexture);

            Tensor inputTensor = new Tensor(inputTesorRenderTexture, channels: 3);


            // Execute the model with the input tensor
            worker.Execute(inputTensor);

            // Retrieve the output tensor
            Tensor outputTensor = worker.PeekOutput();

            string sss = "(";
            outputTensor.shape.ToArray().Each(_ => sss + ", " + _);

            Debug.Log("shape = " + sss + ")");


            // Assuming `output` is the tensor with shape (1, 1, 6, 25200)
            float[] data = outputTensor.ToReadOnlyArray(); // Flatten the tensor into a readable array

            var tree = FaceChopped.DekkuTree(outputTensor);

            var read = FaceChopped.ReadTensor(
                DetectionThreshold,
                NmsThreshold,
                ConfidenceThreshold,
                inputTesorRenderTexture,
                data)
                    .Each(_ => _.Rectangle)
                    .Each(r => new Rect(r.X, r.Y, r.Width, r.Height))
                    .ToList();

            var detectionResults = tree;// (flip = !flip) ? tree : read;

            Debug.Log("found " + detectionResults.Count + " faces");

            if (null == outputTexture2D)
                outputMaterial.mainTexture = outputTexture2D = outputTexture2D = new Texture2D(inputTesorRenderTexture.width, inputTesorRenderTexture.height);

            // fill it randomly
            outputTexture2D.Confetti(
                detectionResults);

            // push the changes to the GPU
            outputTexture2D.Apply();


            // Dispose of the input tensor to free resources
            inputTensor.Dispose();
            outputTensor.Dispose();
        }
    }
    bool flip;

    public Material outputMaterial;
    private void OnDestroy()
    {
        worker?.Dispose();
    }
}
