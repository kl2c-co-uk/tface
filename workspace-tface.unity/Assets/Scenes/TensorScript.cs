
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;


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
    void Update()
    {
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
    }
    private void OnDestroy()
    {
        worker?.Dispose();
    }
}
