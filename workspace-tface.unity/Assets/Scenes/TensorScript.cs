
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System.Drawing;


public static class E
{
    public static int[] Range(this int i)
    {
        var a = new int[i];
        for (i = 0; i < a.Length; i++)
            a[i] = i;
        return a;
    }

    public static void Each<I>(this IEnumerable<I> i, System.Action<I> f)
    {
        foreach (var e in i)
            f(e);
    }
}

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

    }

    public WebcamDisplay webcamDisplay;
    public RenderTexture inputTesorRenderTexture;
    void Update()
    {


#if true
        var webcamTexture = webcamDisplay.webcamTexture;
        // Check if the webcam has provided new data
        if (webcamTexture.didUpdateThisFrame)
        {
            // Convert the webcam texture to a Tensor
            Graphics.Blit(source: webcamTexture, dest: inputTesorRenderTexture);

            Tensor inputTensor = new Tensor(inputTesorRenderTexture, channels: 3);


            // Execute the model with the input tensor
            worker.Execute(inputTensor);

            // Retrieve the output tensor (if needed)
            Tensor outputTensor = worker.PeekOutput();

            Debug.Log(outputTensor);
            Debug.Log(outputTensor);
            Debug.Log(outputTensor);

            var o0 = outputTensor[0, 0, 0, 0];
            var o1 = outputTensor[0, 0, 0, 1];

            cube1.enabled = !(0 >= ((int)o0));
            cube2.enabled = !(0 >= ((int)o1));

            // Dispose of the input tensor to free resources
            inputTensor.Dispose();
            outputTensor.Dispose();
        }
#else

        Debug.Assert(1 == runtimeModel.inputs.Count); // i can assume trhis?

        var batchSize = runtimeModel.inputs[0].shape[4];
        var width = runtimeModel.inputs[0].shape[5];
        var height = runtimeModel.inputs[0].shape[6];
        var channels = runtimeModel.inputs[0].shape[7];

        Debug.Assert(1 == batchSize);
        Debug.Assert(3 == channels);

        // create a tensor
        Tensor inputTensor = new Tensor(1, height, width, channels);

        // fill it with garbatge
        //foreach (var c in channels.Range())
        //    foreach (var w in width.Range())
        //        foreach (var h in height.Range())
        //            inputTensor[0, c, w, h] = Random.value;


        // Execute the model
        worker.Execute(inputTensor);

        // Get the output
        Tensor outputTensor = worker.PeekOutput();

        //Debug.Log("outputTensor.batch = " + outputTensor.batch);
        //Debug.Log("outputTensor.channels = " + outputTensor.channels);
        //Debug.Log("outputTensor.width = " + outputTensor.width);
        //Debug.Log("outputTensor.height = " + outputTensor.height);

        var o0 = outputTensor[0, 0, 0, 0];
        var o1 = outputTensor[0, 0, 0, 1];

        cube1.enabled = !(0 >= ((int)o0));
        cube2.enabled = !(0 >= ((int)o1));

        inputTensor.Dispose();
        outputTensor.Dispose();
#endif
    }
    private void OnDestroy()
    {
        worker?.Dispose();
    }
}
