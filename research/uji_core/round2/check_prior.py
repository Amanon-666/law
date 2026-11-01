"""Check the deployable historical weighting, including partial and empty H."""
import numpy as np
from refi import fit_map

def check():
    h0=dict(rss=np.array([[-45,100],[-55,100],[-90,-55]]),xy=np.array([[0,0],[0,0],[10,0]]))
    h1=dict(rss=np.array([[-90,100],[-44,-74]]),xy=np.array([[0,0],[20,0]]))
    s=dict(rss=np.array([[-71,-65]]),xy=np.array([[12,0]]))
    empty={k:v[:0] for k,v in h0.items()}
    base=fit_map([h0,h1],s)
    tied=fit_map([h0,h1],s,floor_gaps=[1,-1])
    np.testing.assert_array_equal(base.prototypes,tied.prototypes)
    expected=fit_map([h0,h1],s,floor_gaps=[1,3])
    for hs,gaps in [([h1,h0],[3,1]),([empty,h0,h1],[2,1,3]),([h0,h1],[-1,-3])]:
        test=fit_map(hs,s,floor_gaps=gaps)
        np.testing.assert_allclose(test.prototypes,expected.prototypes,atol=1e-10)
    try:
        fit_map([h0,h1],s,floor_gaps=[1])
    except ValueError:
        pass
    else:
        raise AssertionError('Incorrect history metadata accepted')
    print('Floor-weighted prior checks passed')


if __name__=='__main__':check()
