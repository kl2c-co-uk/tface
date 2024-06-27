using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SpinCube : MonoBehaviour
{
    void Update()
    {
        transform.Rotate(Vector3.up * 14.3f * Time.deltaTime);
    }
}
