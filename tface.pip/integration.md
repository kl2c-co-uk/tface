
Intgertating the YOLO network was fairly tedious but straightfoward.
    Needed to add the Unity dependencies and the standalone tface thing, then, copy and paste the FaceONNX node to use YOLO.
    Lazily initialising all the extyra textures was weird, but, solved by just building a delegate object and using that.
    Once done - the thing seemed  to work naievely.

TFace Package URL (for Unity Package Manager); `git@github.com:kl2c-co-uk/tface.git?path=workspace-tface.unity/Packages/uk.co.kl2c.tface#default`

Issues identifeid which relate to TFace are;
- i was not expecteting that I'd need a MonoBehaviour (or whatnot) to carry the asset/NNModel ref into the system.
    - unexpected; i don't think that this really matters though
- [ ] faces are slightly off; seem to have bottom-corner where center should be
    - unexpected; check the bubble-drawer
- [ ] cartoon faces aren't reliable
    - expected; need to add cartoon face dataset
- [ ] possible issue finding certain styles of human faces
    - unexpected; expand the training set

Other than the faces being offset - no real surprises.
Performance is excellent which is unexpected and welcome.
