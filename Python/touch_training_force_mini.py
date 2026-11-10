"""Compact force numbers and signed arrows, no hardware access in paint."""
from PyQt5 import QtCore,QtGui,QtWidgets
class TrainingForceMini(QtWidgets.QWidget):
 AXIS_COLORS={'Fx':(232,76,91),'Fy':(35,177,111),'Fz':(58,124,246)}
 def __init__(self,parent=None,title='PX6D实测力',**kwargs):
  super().__init__(parent);self.title=title;self.force_xyz=(None,)*3;self.connected=False;self.moments=None;self.show_connection_status=False
  self.setMinimumSize(180,75);self.setMaximumHeight(80)
  self.zero_button=QtWidgets.QPushButton('力归零',self)
  self.zero_button.setToolTip('主机去皮：采集与训练部署页共享实测力零偏；不改变模型预测或波长基准')
  self.zero_button.setSizePolicy(QtWidgets.QSizePolicy.Minimum,QtWidgets.QSizePolicy.Fixed)
 def resizeEvent(self,event):
  b=self.zero_button;fm=b.fontMetrics();b.setFixedSize(max(b.sizeHint().width(),fm.horizontalAdvance(b.text())+24),max(b.sizeHint().height(),fm.height()+12));b.move(max(0,self.width()-b.width()-2),0);super().resizeEvent(event)
 def set_force(self,*values,connected=False,**kwargs):self.force_xyz=values[:3];self.connected=connected;self.moments=kwargs.get("moments");self.update()
 def paintEvent(self,event):
  p=QtGui.QPainter(self);p.setRenderHint(QtGui.QPainter.Antialiasing);p.setPen(QtGui.QColor('#53627B'));p.drawText(4,16,self.title+' (N)')
  for i,(axis,color) in enumerate(self.AXIS_COLORS.items()):
   x=4+i*self.width()/3;v=self.force_xyz[i];p.setPen(QtGui.QColor(*color));p.drawText(QtCore.QRectF(x,23,self.width()/3,16),QtCore.Qt.AlignLeft,axis);p.drawText(QtCore.QRectF(x,39,self.width()/3,18),QtCore.Qt.AlignLeft,('—' if v is None else f'{v:+.3f}'))
   origin=QtCore.QPointF(x+20,62);p.setPen(QtGui.QPen(QtGui.QColor(*color),1,QtCore.Qt.DashLine));p.drawLine(origin+QtCore.QPointF(-16,0),origin+QtCore.QPointF(16,0))
   if self.connected and v is not None and abs(v)>1e-9:
    d=1 if v>0 else -1;tip=origin+QtCore.QPointF(d*18,-5);p.setPen(QtGui.QPen(QtGui.QColor(*color),2));p.drawLine(origin,tip);p.drawLine(tip,tip+QtCore.QPointF(-d*6,-3));p.drawLine(tip,tip+QtCore.QPointF(-d*6,4))
