using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TensorFlow;

public class TensorScript : MonoBehaviour
{
    private TFGraph graph;
    private TFSession session;
    //public TextAsset model;
    void Start()
    {
        // Load the frozen model
        TextAsset model = Resources.Load<TextAsset>("face_detector");
        Debug.Assert(null != model, "loading the asset by name did not work");

        // Create a new graph
        graph = new TFGraph();
        graph.Import(model.bytes);

        // Create a new session
        session = new TFSession(graph);

        // Example: Predicting with the model
        float[] input_data = new float[32]; // Your input data
        var runner = session.GetRunner();
        runner.AddInput(graph["input_node_name"][0], input_data);
        runner.Fetch(graph["output_node_name"][0]);
        var output = runner.Run();

        // Retrieve output
        float[] result = ((float[][])output[0].GetValue(jagged: true))[0];

        // Use the result
        Debug.Log("Model prediction: " + result[0]);
    }

    // Update is called once per frame
    void OnDestroy()
    {
        session.CloseSession();
        graph.Dispose();
    }
}
