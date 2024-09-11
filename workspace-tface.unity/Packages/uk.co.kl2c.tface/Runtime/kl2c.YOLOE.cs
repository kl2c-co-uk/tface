using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using Unity.VisualScripting;
using UnityEngine;
using System.Linq;
using System.Threading.Tasks;
namespace kl2c
{
	/// <summary>
	/// code and extensions for working with the/a YOLO models
	/// </summary>
	public static class YOLOE
	{

		/// <summary>
		/// redundant - use ForEach from LinQ
		/// </summary>
		public static int[] Range(this int i)
		{
			var a = new int[i];
			for (i = 0; i < a.Length; i++)
				a[i] = i;
			return a;
		}


		public static System.Func<System.Func<O, I, O>, O> Fold<O, I>(this IEnumerable<I> i, O o)
		{
			return f =>
			{
				foreach (var e in i)
					o = f(o, e);
				return o;
			};
		}


		public static O[] Fork<I, O>(this IEnumerable<I> i, System.Func<I, O> f)
		{
			var input = i.ToArray();

			var output = new O[input.Length];

			Parallel.ForEach(Enumerable.Range(0, output.Length), i =>
			{
				output[i] = f(input[i]);
			});

			return output;
		}

		/// <summary>
		/// redundant - use ForEach from LinQ
		/// </summary>
		/// <typeparam name="I"></typeparam>
		/// <param name="i"></param>
		/// <param name="f"></param>
		public static void Each<I>(this IEnumerable<I> i, System.Action<I> f)
		{
			foreach (var e in i)
				f(e);
		}

		public static IEnumerable<I> Drop<I>(this IEnumerable<I> l, int i)
		{

			foreach (var e in l)
				if (i > 0)
					i--;
				else
					yield return e;
		}

		/// <summary>
		/// redundant - use Select from LinQ
		/// </summary>
		/// <typeparam name="I"></typeparam>
		/// <param name="i"></param>
		/// <param name="f"></param>
		public static IEnumerable<O> Each<I, O>(this IEnumerable<I> i, System.Func<I, O> f)
		{
			return i.Select(f);
		}

		public static void Fill(this Texture2D target, UnityEngine.Color colour)
		{
			target.SetPixels(new UnityEngine.Color[target.width * target.height].Each(_ => colour).ToArray());
		}


		public static void FaceOvals(this Texture2D target, IEnumerable<Rect> patches)
		{
			var clear = UnityEngine.Color.black;
			var solid = UnityEngine.Color.white;
			clear.a = 0;
			solid.a = 1;
			target.FaceOvals(patches, clear, solid);
		}


		public static void FaceOvals(this Texture2D target, IEnumerable<Rect> patches, UnityEngine.Color clear, UnityEngine.Color solid)
		{
			patches.Each(rectangle =>
			{
				float w = 2.0f / (float)rectangle.width;
				float h = 2.0f / (float)rectangle.height;
				for (int x = (int)rectangle.xMin; x < (int)rectangle.xMax; ++x)
					for (int y = (int)rectangle.yMin; y < (int)rectangle.yMax; ++y)
					{
						float i = w * (rectangle.center.x - x);
						float j = h * (rectangle.center.y - y);

						float r2 = (i * i) + (j * j);

						if (r2 > 1)
							continue;

						var c = UnityEngine.Color.Lerp(
							UnityEngine.Color.white,
							UnityEngine.Color.black,
							r2);

						var o = target.GetPixel(x, y);

						if (o.a < c.a)
							target.SetPixel(x, y, c);
					}
			});
		}



		/// <summary>
		/// fill in faces with random boxes. used fur debugging (sorry)
		/// </summary>
		public static void Confetti(this Texture2D target, IEnumerable<Rect> patches, System.Random random = null)
		{
			if (null == random)
			{
				var p = patches.ToArray();
				random = new System.Random(Seed: p.Length);
				patches = p;
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

			patches.Each(rectangle =>
			{
				var colour = colours[random.Next(0, colours.Length)];
				for (int x = (int)rectangle.xMin; x < (int)rectangle.xMax; ++x)
					for (int y = (int)rectangle.yMin; y < (int)rectangle.yMax; ++y)
						target.SetPixel(x, y, colour);
			});
		}
	}
}
