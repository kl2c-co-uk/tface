
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


        {

            Debug.Assert(1 == runtimeModel.inputs.Count); // i can assume trhis?

            //// Inspect the model to get input dimensions
            //Model.Input firstInput = runtimeModel.inputs[0]; // Assuming the model has at least one input
            //string inputName = firstInput.name;
            //int[] inputShape = firstInput.shape;

            //Debug.Log($"Input Name: {inputName}");
            //Debug.Log($"Input Shape: {string.Join(", ", inputShape)}");


            //// Inspect the model to get input dimensions
            //foreach (var input in runtimeModel.inputs)
            //{
            //    string inputName = input.name;
            //    int[] inputShape = input.shape;

            //    Debug.Log($"Input Name: {inputName}");
            //    Debug.Log($"Input Shape: {string.Join(", ", inputShape)}");
            //}





            var batchSize = runtimeModel.inputs[0].shape[4];
            var width = runtimeModel.inputs[0].shape[5];
            var height = runtimeModel.inputs[0].shape[6];
            var channels = runtimeModel.inputs[0].shape[7];

            Debug.Assert(1 == batchSize);

            // fill it with garbatge
            Tensor inputTensor = new Tensor(1, channels, width, height);

            foreach (var c in channels.Range())
                foreach (var w in width.Range())
                    foreach (var h in height.Range())
                        inputTensor[0, c, w, h] = Random.value;


            // Execute the model
            worker.Execute(inputTensor);

            if (true) throw new System.Exception("??? - now read from result?");

        }


        if (true) throw new System.Exception("??? - prepare proper inputs");
        {
            // Example: Prepare input data
            Tensor inputTensor = new Tensor(1, 32); // Adjust dimensions as needed
            for (int i = 0; i < 32; i++)
            {
                inputTensor[0, i] = 0.0f; // Replace with your input data
            }

            // Execute the model
            worker.Execute(inputTensor);

            // Get the output
            Tensor outputTensor = worker.PeekOutput();
            float outputValue = outputTensor[0]; // Adjust based on output dimensions

            // Use the result
            Debug.Log("Model prediction: " + outputValue);

            // Clean up
            inputTensor.Dispose();
            outputTensor.Dispose();
        }

        worker?.Dispose();
    }
}
