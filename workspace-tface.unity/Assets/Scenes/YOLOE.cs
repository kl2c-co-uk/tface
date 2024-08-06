using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using Unity.VisualScripting;
using UnityEngine;
using System.Linq;

/// <summary>
/// code and extensions for working with the/a YOLO models
/// </summary>
public static class YOLOE
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

    public static void Confetti(this Texture2D target, IEnumerable<Rect> patches)
    {
        var p = patches.AsReadOnlyList();
        Confetti(target, p, new System.Random(Seed: p.Count));
    }

    /// <summary>
    /// fill in faces with random boxes. used fur debugging (sorry)
    /// </summary>
    public static void Confetti(this Texture2D target, IEnumerable<Rect> patches, System.Random random)
    {
        target.SetPixels(new UnityEngine.Color[target.width * target.height].Each(_ => UnityEngine.Color.black).ToArray());
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

        patches.Each(rectangle =>
        {
            var colour = colours[random.Next(0, colours.Length)];
            for (int x = (int)rectangle.xMin; x < (int)rectangle.xMax; ++x)
                for (int y = (int)rectangle.yMin; y < (int)rectangle.yMax; ++y)
                    target.SetPixel(x, y, colour);
        });
    }
}