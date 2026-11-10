import unittest
import numpy as np
class Single26Tests(unittest.TestCase):
 def test_official_head_is_exclusive_26_and_legacy_stays25(self):
  import torch
  from touch_tslib45 import TSLib45Task
  c=dict(architecture='tslib45_mixer',width=64,layers=3,heads=4,dropout=.1,position_mode='single_label26_v1')
  self.assertEqual(tuple(TSLib45Task(c,'position')(torch.zeros(2,45)).shape),(2,26))
  c.pop('position_mode');self.assertEqual(tuple(TSLib45Task(c,'position')(torch.zeros(2,45)).shape),(2,25))
 def test_softmax_decode_all_classes_and_empty(self):
  from touch_single26_contract import decode_single26
  for i in range(26):
   p=np.full(26,.01);p[i]=.75;p/=p.sum();r=decode_single26(p)
   self.assertEqual(r['position'],i);self.assertEqual(r['position_labels'],[i] if i else [])
   self.assertEqual(len(r['probabilities']),25);self.assertEqual(sum(r['position_multihot']),int(i>0))
 def test_multi_is_excluded_not_relabelled(self):
  from touch_single26_contract import class_from_multihot
  self.assertEqual(class_from_multihot(np.zeros(25)),0)
  y=np.zeros(25);y[24]=1;self.assertEqual(class_from_multihot(y),25)
  y[0]=1
  with self.assertRaises(ValueError):class_from_multihot(y)
if __name__=='__main__':unittest.main()
