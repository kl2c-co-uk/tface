
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

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

    void OnDestroy()
    {
        worker?.Dispose();
    }
}
