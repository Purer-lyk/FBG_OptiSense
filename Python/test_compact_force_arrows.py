"""Compact force plot pixel regression; no sensor/hardware startup."""
import os,sys,unittest
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=str(Path(sys.prefix)/'Lib/site-packages/PyQt5/Qt5/plugins/platforms')
from PyQt5 import QtWidgets,QtCore
from manual_stress_capture import ForceVectorWidget
class CompactForceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
 def test_compact_plot_has_actual_rgb_arrows_below_cards(self):
  w=ForceVectorWidget(compact=True);w.resize(280,390);w.set_force(1,-2,3);w.show();self.app.processEvents()
  image=w.grab().toImage()
  for rgb in w.AXIS_COLORS.values():
   count=sum(all(abs(image.pixelColor(x,y).getRgb()[i]-rgb[i])<15 for i in range(3)) for y in range(245,350) for x in range(10,w.width()-10))
   self.assertGreater(count,35,'compact plot is missing colored arrow pixels')
  w.close()
 def test_signed_projection_preserves_values_and_clears_unavailable(self):
  w=ForceVectorWidget(compact=True);w.resize(280,330)
  for values in ((1.,-2.,3.),(-.001,-.040,-.096),(1e8,-2e8,3e8)):
   w.set_force(*values);origin,reach,vectors=w._compact_force_geometry()
   self.assertEqual(w.force_xyz,values)
   for (_,d,t),v in zip(vectors,values):
    delta=t-origin
    self.assertGreater((delta.x()*d.x()+delta.y()*d.y())*v,0)
  w.set_force(0.,0.,0.)
  self.assertTrue(all(t is None for _,_,t in w._compact_force_geometry()[2]))
  w.set_force(None,None,None)
  self.assertFalse(w.connected)
  self.assertTrue(all(t is None for _,_,t in w._compact_force_geometry()[2]))
  w.close()
if __name__=='__main__':unittest.main()
