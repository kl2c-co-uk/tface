using System;
using System.Drawing;
using System.Threading.Tasks;

namespace UMapx.Core
{
    /// <summary>
    /// Uses to implement standard algebraic operations on matrices and vectors.
    /// </summary>
    public static class Matrice
    {
        /// <summary>
        /// Gets the value of the maximum element of the vector.
        /// </summary>
        /// <param name="v">Array</param>
        /// <param name="index">Max index</param>
        /// <returns>float precision floating point number</returns>
        public static float Max(this float[] v, out int index)
        {
            int length = v.Length;
            float maximum = float.MinValue;
            float c;
            index = 0;

            for (int i = 0; i < length; i++)
            {
                c = v[i];

                if (c > maximum)
                {
                    maximum = c;
                    index = i;
                }
            }
            return maximum;
        }

    }
}
