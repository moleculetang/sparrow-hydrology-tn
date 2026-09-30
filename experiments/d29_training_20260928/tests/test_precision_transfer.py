import unittest
import numpy as np
from d29_platform.precision_transfer import transfer_expansion,represented_balance

class TransferTests(unittest.TestCase):
    def test_large_stock_conservative_expansion(self):
        rng=np.random.default_rng(1729)
        high=rng.uniform(0,1e10,(7,13,5));low=np.zeros_like(high)
        matrix=rng.uniform(0,1,(7,13,13));matrix/=matrix.sum(-1,keepdims=True)
        for _ in range(3):
            new,tail,change=transfer_expansion(high,low,matrix)
            self.assertLess(represented_balance(new,tail,high,low),1e-6)
            self.assertLess(change,1e-14)
            self.assertTrue((new>=0).all())
            high,low=new,tail

    def test_identity_and_known_split(self):
        high=np.array([[[3.,5.],[7.,11.]]]);low=np.zeros_like(high)
        a,b,c=transfer_expansion(high,low,np.eye(2)[None])
        np.testing.assert_array_equal(a,high);np.testing.assert_array_equal(b,low)
        matrix=np.array([[[.25,.75],[.5,.5]]])
        a,b,c=transfer_expansion(high,low,matrix)
        np.testing.assert_array_equal(a,np.array([[[4.25,6.75],[5.75,9.25]]]))

if __name__=='__main__':unittest.main()
