import visualization
import xyPlot
import connectorBehavior
import odbAccess
o1 = session.openOdb(name='C:/Users/osama/OneDrive/Desktop/abaqus2/o/mmmooo.odb')
xy_result = session.XYDataFromHistory(name='aa', odb=o1, outputVariableName='Reaction force: RF3 PI: rootAssembly Node 2 in NSET SET-3', steps=('Step-1', ), )
x0 = session.xyDataObjects['aa']
session.writeXYReport(fileName='C:/Users/osama/OneDrive/Desktop/abaqus2/o/abaqusoo.rpt', xyData=(x0, ))
