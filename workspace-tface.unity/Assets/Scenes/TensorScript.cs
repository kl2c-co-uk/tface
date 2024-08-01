
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System.IO;
using System.Drawing;
using System.Net.WebSockets;
using System.Linq;

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

    public static IEnumerable<O> Each<I, O>(this IEnumerable<I> i, System.Func<I, O> f)
    {
        foreach (var e in i)
            yield return f(e);
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

            // is this wrong?
            // ... well ... then it's probably fine anyway

            //int count = data.Length / 6;
            //Debug.Assert((25200 * 6) == data.Length);

            //using (StreamWriter writer = new StreamWriter("detekt.csv"))
            //{
            //    writer.WriteLine("{i}, {x_center}, {y_center}, {width}, {height}, {confidence_i_found_a_thing}, {confidence_its_a_face},");

            //    for (int i = 0; i < 25200; i++)
            //    {
            //        int baseIndex = i * 6;
            //        float x_center = data[baseIndex];
            //        float y_center = data[baseIndex + 1];
            //        float width = data[baseIndex + 2];
            //        float height = data[baseIndex + 3];
            //        float confidence_i_found_a_thing = data[baseIndex + 4];
            //        float confidence_its_a_face = data[baseIndex + 5]; // Assuming a single class or a score


            //        writer.WriteLine($"{i}, {x_center}, {y_center}, {width}, {height}, {confidence_i_found_a_thing}, {confidence_its_a_face},");
            //    }
            //}

            using (StreamWriter writer = new StreamWriter("face_onnx_way.csv"))
            {
                writer.Write("{rectangle.Left}, {rectangle.Top}, {rectangle.Right}, {rectangle.Top},\n");

                // https://github.com/FaceONNX/FaceONNX/blob/main/netstandard/FaceONNX/face/classes/FaceDetector.cs#L108


                var Labels = new string[] { "face" };
                var size = new Size(inputTesorRenderTexture.width, inputTesorRenderTexture.height);
                var width = size.Width; //  image[0].GetLength(1);
                var height = size.Height; //  image[0].GetLength(0);

                // yolo params
                var yoloSquare = 15;
                var classes = Labels.Length;
                var count = classes + yoloSquare;

                // post-processing
                var vector = data; // results[0].AsTensor<float>().ToArray();
                var length = vector.Length / count;
                var predictions = new float[length][];

                for (int i = 0; i < length; i++)
                {
                    var prediction = new float[count];

                    for (int j = 0; j < count; j++)
                        prediction[j] = vector[i * count + j];

                    predictions[i] = prediction;
                }

                var list = new List<float[]>();

                // seivining results
                for (int i = 0; i < length; i++)
                {
                    var prediction = predictions[i];

                    if (prediction[4] > DetectionThreshold)
                    {
                        var a = prediction[0];
                        var b = prediction[1];
                        var c = prediction[2];
                        var d = prediction[3];

                        prediction[0] = a - c / 2;
                        prediction[1] = b - d / 2;
                        prediction[2] = a + c / 2;
                        prediction[3] = b + d / 2;

                        //for (int j = yoloSquare; j < prediction.Length; j++)
                        //{
                        //    prediction[j] *= prediction[4];
                        //}

                        list.Add(prediction);
                    }
                }

                // non-max suppression
                list = FaceONNX.NonMaxSuppressionExensions.AgnosticNMSFiltration(list, NmsThreshold);

                // perform
                predictions = list.ToArray();
                length = predictions.Length;

                // backward transform
                var k0 = (float)size.Width / width;
                var k1 = (float)size.Height / height;
                float gain = Mathf.Min(k0, k1);
                float p0 = (size.Width - width * gain) / 2;
                float p1 = (size.Height - height * gain) / 2;

                // collect results
                var detectionResults = new List<FaceONNX.FaceDetectionResult>();

                for (int i = 0; i < length; i++)
                {
                    var prediction = predictions[i];
                    var labels = new float[classes];

                    for (int j = 0; j < classes; j++)
                    {
                        labels[j] = prediction[j + yoloSquare];
                    }

                    var max = UMapx.Core.Matrice.Max(labels, out int argmax);

                    if (max > ConfidenceThreshold)
                    {
                        var rectangle = Rectangle.FromLTRB(
                            (int)((prediction[0] - p0) / gain),
                            (int)((prediction[1] - p1) / gain),
                            (int)((prediction[2] - p0) / gain),
                            (int)((prediction[3] - p1) / gain));

                        var points = new Point[5];

                        for (int j = 0; j < 5; j++)
                        {
                            points[j] = new Point
                            {
                                X = (int)((prediction[5 + 2 * j + 0] - p0) / gain),
                                Y = (int)((prediction[5 + 2 * j + 1] - p1) / gain)
                            };
                        }

                        var landmarks = new FaceONNX.Face5Landmarks(points);

                        writer.Write($"{rectangle.Left}, {rectangle.Top}, {rectangle.Right}, {rectangle.Top},\n");

                        detectionResults.Add(new FaceONNX.FaceDetectionResult
                        {
                            Rectangle = rectangle,
                            Id = argmax,
                            Score = max,
                            Points = landmarks
                        });
                    }
                }

                Debug.Log("found " + detectionResults.Count + " faces");

                {
                    // clear the output tteure
                    {
                        if (null == outputTexture2D)
                            outputMaterial.mainTexture = outputTexture2D = outputTexture2D = new Texture2D(inputTesorRenderTexture.width, inputTesorRenderTexture.height);


                        outputTexture2D.SetPixels(new UnityEngine.Color[outputTexture2D.width * outputTexture2D.height].Each(_ => UnityEngine.Color.black).ToArray());
                    }

                    var colours = new UnityEngine.Color[]
                    {
                        UnityEngine.Color.blue,
                        UnityEngine.Color.cyan,
                        UnityEngine.Color.gray,
                        UnityEngine.Color.green,
                        UnityEngine.Color.grey,
                        UnityEngine.Color.magenta,
                        UnityEngine.Color.red,
                        UnityEngine.Color.white,
                        UnityEngine.Color.yellow,
                    };

                    var random = new System.Random(Seed: detectionResults.Count);

                    // our old thing was just the Rect for some reason
                    foreach (var rect in detectionResults.Each(_ => _.Rectangle))
                    {
                        var colour = colours[random.Next(0, colours.Length)];
                        for (int x = rect.Left; x < rect.Right; ++x)
                        {
                            for (int y = rect.Top; y < rect.Bottom; ++y)
                            {
                                outputTexture2D.SetPixel(x, y, colour);
                            }
                        }
                    }

                    outputTexture2D.Apply();
                }
            }


            // Dispose of the input tensor to free resources
            inputTensor.Dispose();
            outputTensor.Dispose();
        }
    }

    public Material outputMaterial;
    private void OnDestroy()
    {
        worker?.Dispose();
    }
}
