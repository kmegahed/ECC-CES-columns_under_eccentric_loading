
B=300.0;H=600.0;lam=35.0;fysteel=230.0;fe=25.0;fr=250.0;d_steel=0.3;ro_rebar=0.01;tf_tw=1.50;cH=60;cB=50;eccXY=600
Astr=50;S_space=1000*Astr*2/min(B,H);fstr=235.0;As_4=ro_rebar*B*H/4.0;cover=30;
L=lam*B;H2=H/2.0;B2=B/2.0;hs2=(H-2.0*cH)/2.0;bf2=(B-2.0*cB)/2.0;
A_steel_4=(4.0*As_4*fr+0.85*B*H*0.97*fe)*d_steel/fysteel/4.0;tw2=A_steel_4/((bf2-4.0)*tf_tw+hs2);tf=(A_steel_4-hs2*tw2)/(bf2-tw2)
axis=1;
#name B_H_bf2_hs2_tf_tw2_lam_fs_fy_d_ro_tfw_cH_cB
#hs2=120.0/2.0;bf2=120.0/2.0;tw2=5.5/2;tf=5.9;H2=160.0/2.0;B2=160.0/2.0;L=500;fy=752;fe=90.85;As=0.001;fs=100;Astr=0.001;fstr=100;cover=10;S_space=100;
import numpy as np
f_fc = 0.85;Es = 210000; Ec = 22000 * ((fe + 8.) / 10.)**0.3; Ar = ro_rebar * B * H;  As = ((hs2 * 2.) * (bf2 * 2.) - (hs2 * 2. - 2. * tf) * (bf2 * 2. - 2. * tw2));Ac = H * B - Ar - As
I_S_X = ((hs2 * 2.)**3. * (bf2 * 2.) - (hs2 * 2. - 2. * tf)**3 * (bf2 * 2. - 2. * tw2)) / 12.0; I_REBAR_X = Ar * (H - 2.0 * cover)**2. / 4.0; 
EI1 = (I_S_X + I_REBAR_X) * Es + (H**3. * B/12.0 - I_S_X - I_REBAR_X) * Ec * 0.6

I_S_Y = ((bf2 * 2.)**3. * tf * 2.0 + (hs2 * 2. - 2. * tf) * (2. * tw2)**3) / 12.0;    I_REBAR_Y = Ar * (B - 2.0 * cover)**2. / 4.0
EI2 = (I_S_Y + I_REBAR_Y) * Es + (H * B**3./12.0 - I_S_Y - I_REBAR_Y) * Ec * 0.6
EI = min(EI1, EI2);    pc = Ac * fe * f_fc; pa = pc + As * fysteel + Ar * fr;   Ncr = (np.pi**2. * EI) / (L * L); slender = (pa / Ncr)**0.5
mod1='Model-1'
import os
os.chdir(r"C:\Users\osama\OneDrive\Desktop\abaqus2\o")
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
mesh_size=max(B/6.0,tw2*8.0)

from abaqus import *
from abaqusConstants import *
import regionToolset
import __main__
import section
import regionToolset
import part
import material
import assembly
import step
import interaction
import load
import mesh
import job
import sketch
import visualization
import xyPlot
import connectorBehavior
import odbAccess
from operator import add
import meshEdit

mdb.Model(name=mod1, modelType=STANDARD_EXPLICIT)
s = mdb.models[mod1].ConstrainedSketch(name='__profile__', sheetSize=200.0)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)
s.rectangle(point1=(-B2, -H2), point2=(B2, H2))
p = mdb.models[mod1].Part(name='Part-1', dimensionality=THREE_D, type=DEFORMABLE_BODY)
p = mdb.models[mod1].parts['Part-1']
p.BaseSolidExtrude(sketch=s, depth=L)
p = mdb.models[mod1].parts['Part-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
del mdb.models[mod1].sketches['__profile__']


p = mdb.models[mod1].parts['Part-1']
f1= p.faces.getByBoundingBox(-B2-100-1.0, -H2-100-1.0, -1.0,          B2+100+1.0, H2+100+1.0,0.0001)
e1= p.edges.getByBoundingBox(B2-1.0, -H2-100-1.0, -1.0,          B2+100+1.0, H2+100+1.0,0.0001)
#t = p.MakeSketchTransform(sketchPlane=f1, sketchUpEdge=e1, sketchPlaneSide=SIDE1, origin=(0, 0, 0))
s1 = mdb.models[mod1].ConstrainedSketch(name='__profile__',     sheetSize=241.86)
s1.setPrimaryObject(option=SUPERIMPOSE)
p = mdb.models[mod1].parts['Part-1']
p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)
s1.Line(point1=(-bf2, -hs2), point2=(bf2, -hs2))
s1.Line(point1=(bf2, -hs2), point2=(bf2, -hs2+tf))
s1.Line(point1=(bf2, -hs2+tf), point2=(tw2, -hs2+tf))
s1.Line(point1=(tw2, -hs2+tf), point2=(tw2, hs2-tf))
s1.Line(point1=(tw2, hs2-tf), point2=(bf2, hs2-tf))
s1.Line(point1=(bf2, hs2-tf), point2=(bf2, hs2))
s1.Line(point1=(bf2, hs2), point2=(-bf2, hs2))
s1.Line(point1=(-bf2, hs2), point2=(-bf2, hs2-tf))
s1.Line(point1=(-bf2, hs2-tf), point2=(-tw2, hs2-tf))
s1.Line(point1=(-tw2, hs2-tf), point2=(-tw2, -hs2+tf))
s1.Line(point1=(-tw2, -hs2+tf), point2=(-bf2, -hs2+tf))
s1.Line(point1=(-bf2, -hs2+tf), point2=(-bf2, -hs2))

p = mdb.models[mod1].parts['Part-1']
p.PartitionFaceBySketch(faces=f1, sketch=s1)
s1.unsetPrimaryObject()
del mdb.models[mod1].sketches['__profile__']
p = mdb.models[mod1].parts['Part-1']

c = p.cells
e= p.edges.getByBoundingBox(-bf2-1.0, -hs2-1.0, -1.0,     bf2+1.0, hs2+1.0,1.0)
e1= p.edges.getByBoundingBox(B2-1.0, H2-1.0, -1.0,   B2+1.0, H2+1.0,L+0.1)
p.PartitionCellByExtrudeEdge(cells=c, edges=e,line=e1[0], sense=FORWARD)

mdb.models[mod1].Material(name='steel')
mdb.models[mod1].materials['steel'].Elastic(table=((210000.0, 0.3), ))
mdb.models[mod1].materials['steel'].Plastic(table=((fysteel, 0.0), (fysteel*1.02,0.01)))
mdb.models[mod1].HomogeneousSolidSection(name='steel1', material='steel',thickness=None)

mdb.models[mod1].Material(name='rebars')
mdb.models[mod1].materials['rebars'].Elastic(table=((210000.0, 0.3), ))
mdb.models[mod1].materials['rebars'].Plastic(table=((fr, 0.0), (fr*1.02,0.01)))
mdb.models[mod1].TrussSection(name='rebar1', material='rebars', area=As_4)

mdb.models[mod1].Material(name='stirrups')
mdb.models[mod1].materials['stirrups'].Elastic(table=((210000.0, 0.3), ))
mdb.models[mod1].materials['stirrups'].Plastic(table=((fstr, 0.0), (fstr*1.02,0.01)))
mdb.models[mod1].TrussSection(name='stirrup1', material='stirrups', area=Astr)

d_ang,fco,maty,mu,KK=35.0,fe,'concrete',0.35,0.74
mdb.models[mod1].Material(name='concrete')
mdb.models[mod1].Material(name= maty)
xi= mdb.models[mod1].materials[maty]
Ec=1500.0*fe**0.638;f_E=fe/Ec;ec=1.757*f_E;e_av=(0.4*f_E+ec)/2.0;f_av=Ec*e_av*(1.127-0.3175*e_av*Ec/fe)
tably = ((0.4*fe,0),(f_av,e_av-f_av/Ec),(fe,ec-fe/Ec),(0.5*fe,1.5*ec-0.5*fe/Ec),(0.15*fe,5.0*ec-0.15*fe/Ec))
xi.Elastic(table=((Ec, mu), )) 
xi.ConcreteDamagedPlasticity(table=((d_ang,0.1, 1.16, KK, 0.001), ))
xi.concreteDamagedPlasticity.ConcreteCompressionHardening(table=tably)
xi.concreteDamagedPlasticity.ConcreteTensionStiffening(table=((fco/10.0, 0.0),(fco/50.0, 0.01),))#(fco/10.0/1.20, 0.0),(fco/10.0, 0.007), (fco/50.0, 0.01)
mdb.models[mod1].HomogeneousSolidSection(name='conc', material='concrete',  thickness=None)

#cells1 =c.getByBoundingBox(-bf2-1.0, -hs2-1.0, -1.0,     bf2+1.0, hs2+1.0,L+1.0)
#region1 = regionToolset.Region(cells=cells1)

p = mdb.models[mod1].parts['Part-1']
c = p.cells
cells1 =c.findAt(((0.,0.,1.),),)
cells2 =c.findAt(((B2-1.,H2-1.,1.),),)
region1 = regionToolset.Region(cells=cells1)
region2 = regionToolset.Region(cells=cells2)
p.SectionAssignment(region=region1, sectionName='steel1', offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
p.SectionAssignment(region=region2, sectionName='conc', offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)




a1 = mdb.models[mod1].parts['Part-1'];c1= a1.cells
ar1 = a1.DatumAxisByPrincipalAxis(principalAxis=YAXIS).id;ar2 = a1.DatumAxisByPrincipalAxis(principalAxis=XAXIS).id
c1= a1.cells;a1.PartitionCellByPlanePointNormal(point=(-bf2,-hs2,0), normal=a1.datums[ar1], cells=c1[:])
c1= a1.cells;a1.PartitionCellByPlanePointNormal(point=(-bf2,-hs2,0), normal=a1.datums[ar2], cells=c1[:])

c1= a1.cells;a1.PartitionCellByPlanePointNormal(point=(bf2,hs2,0), normal=a1.datums[ar1], cells=c1[:])
c1= a1.cells;a1.PartitionCellByPlanePointNormal(point=(bf2,hs2,0), normal=a1.datums[ar2], cells=c1[:])

c1= a1.cells;a1.PartitionCellByPlanePointNormal(point=(tw2,hs2-tf,0), normal=a1.datums[ar1], cells=c1[:])
c1= a1.cells;a1.PartitionCellByPlanePointNormal(point=(tw2,-hs2+tf,0), normal=a1.datums[ar1], cells=c1[:])

c1= a1.cells;a1.PartitionCellByPlanePointNormal(point=(tw2,hs2-tf,0), normal=a1.datums[ar2], cells=c1[:])
c1= a1.cells;a1.PartitionCellByPlanePointNormal(point=(-tw2,-hs2+tf,0), normal=a1.datums[ar2], cells=c1[:])



p = mdb.models[mod1].parts['Part-1']
p.seedPart(size=mesh_size*3.0, deviationFactor=0.1, minSizeFactor=0.1)
e1= p.edges.getByBoundingBox(-B2-100.0, -H2-100.0, -1.0,     B2+100.0, H2+100.0,1.0)
e2= p.edges.getByBoundingBox(-B2-100.0, -H2-100.0, L-1.0,    B2+100.0, H2+100.0,L+1.0)

p.seedEdgeBySize(edges=e1+e2, size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1, constraint=FINER)
p.generateMesh()



p = mdb.models[mod1].parts['Part-1']
n = p.nodes
n2=[]
for i in n:
    n2.append([i.label,i.coordinates[0]+L/3500.0*(25.0*slender-7.0)*np.sin(i.coordinates[2]/L*pi),i.coordinates[1]+L/3500.0*(25.0*slender-7.0)*np.sin(i.coordinates[2]/L*pi),i.coordinates[2]])

#L/2000.0*(lam-2.0)
#L/200.0*(lam/20.0+1.0)
n3=[]
for i in range(len(n2)):
    n3.append(n2[i][1:4])

p.editNode(nodes=n[0:len(n)], coordinates=tuple(map(tuple,n3)))





s1 = mdb.models[mod1].ConstrainedSketch(name='__profile__',   sheetSize=200.0)
g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
s1.setPrimaryObject(option=STANDALONE)
s1.Line(point1=(0.0, 0.0), point2=(L, 0.0))
p = mdb.models[mod1].Part(name='Part-3', dimensionality=THREE_D, type=DEFORMABLE_BODY)
p = mdb.models[mod1].parts['Part-3']
p.BaseWire(sketch=s1)
s1.unsetPrimaryObject()
p = mdb.models[mod1].parts['Part-3']
del mdb.models[mod1].sketches['__profile__']
e = p.edges
edges = e.getSequenceFromMask(mask=('[#1 ]', ), )
region=regionToolset.Region(edges=edges)
p.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=(0.0, 0.0, -1.0))
edges = e.getSequenceFromMask(mask=('[#1 ]', ), )
region = regionToolset.Region(edges=edges)
p.SectionAssignment(region=region, sectionName='rebar1', offset=0.0,  offsetType=MIDDLE_SURFACE, offsetField='',  thicknessAssignment=FROM_SECTION)

p = mdb.models[mod1].parts['Part-3']
p.seedPart(size=mesh_size*3.0, deviationFactor=0.1, minSizeFactor=0.1)
elemType1 = mesh.ElemType(elemCode=T3D2, elemLibrary=STANDARD)
e = p.edges
edges = e.getSequenceFromMask(mask=('[#1 ]', ), )
pickedRegions =(edges, )
p.setElementType(regions=pickedRegions, elemTypes=(elemType1, ))
p = mdb.models[mod1].parts['Part-3']
p.generateMesh()

n = p.nodes
n2=[]
for i in n:
    n2.append([i.label,i.coordinates[0],i.coordinates[1]+L/3500.0*(25.0*slender-7.0)*np.sin(i.coordinates[0]/L*pi),i.coordinates[2]-L/3500.0*(25.0*slender-7.0)*np.sin(i.coordinates[0]/L*pi)])

n3=[]
for i in range(len(n2)):
    n3.append(n2[i][1:4])

p.editNode(nodes=n[0:len(n)], coordinates=tuple(map(tuple,n3)))




s = mdb.models[mod1].ConstrainedSketch(name='__profile__',     sheetSize=200.0)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)
s.rectangle(point1=(cover-B2, cover-H2), point2=(B2-cover, H2-cover))
p = mdb.models[mod1].Part(name='Part-4', dimensionality=THREE_D,  type=DEFORMABLE_BODY)
p = mdb.models[mod1].parts['Part-4']
p.BaseWire(sketch=s)
s.unsetPrimaryObject()
p = mdb.models[mod1].parts['Part-4']
del mdb.models[mod1].sketches['__profile__']
e = p.edges
edges = e.getSequenceFromMask(mask=('[#f ]', ), )
region = regionToolset.Region(edges=edges)
p = mdb.models[mod1].parts['Part-4']
p.SectionAssignment(region=region, sectionName='stirrup1', offset=0.0,   offsetType=MIDDLE_SURFACE, offsetField='',   thicknessAssignment=FROM_SECTION)


p = mdb.models[mod1].parts['Part-4']
p.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)
elemType1 = mesh.ElemType(elemCode=T3D2, elemLibrary=STANDARD)
e = p.edges
edges = e.getSequenceFromMask(mask=('[#f ]', ), )
pickedRegions =(edges, )
p.setElementType(regions=pickedRegions, elemTypes=(elemType1, ))
p.generateMesh()


import math
n_Rep=math.floor(L/S_space-0.0001)+1;s_first=(L-(n_Rep-1)*S_space)/2.0

for i in range(n_Rep):
    a = mdb.models[mod1].rootAssembly
    a.Instance(name='Part-4-'+str(i+1), part=p, dependent=ON)
    zz=s_first+S_space*i
    a.translate(instanceList=('Part-4-'+str(i+1), ), vector=(L/3500.0*(25.0*slender-7.0)*np.sin(zz/L*pi), L/3500.0*(25.0*slender-7.0)*np.sin(zz/L*pi), zz))

#a.LinearInstancePattern(instanceList=('Part-4-'+str(i), ), direction1=(1.0, 0.0,0.0), direction2=(0.0, 0.0, 1.0), number1=1, number2=int(n_Rep+1), spacing1=30.0, spacing2=S_space)





a = mdb.models[mod1].rootAssembly
a.DatumCsysByDefault(CARTESIAN)
p = mdb.models[mod1].parts['Part-1']
a.Instance(name='Part-1-1', part=p, dependent=ON)
p = mdb.models[mod1].parts['Part-3']
a.Instance(name='Part-3-1', part=p, dependent=ON)
a.rotate(instanceList=('Part-3-1', ), axisPoint=(0.0, 0.0, 0.0),  axisDirection=(0.0, 1.0, 0.0), angle=-90.0)
a.translate(instanceList=('Part-3-1', ), vector=(cover-B2, cover-H2, 0.0))
a.LinearInstancePattern(instanceList=('Part-3-1', ), direction1=(1.0, 0.0, 0.0), direction2=(0.0, 1.0, 0.0), number1=2, number2=2, spacing1=2.0*(B2-cover),  spacing2=2.0*(H2-cover))
ecx=0.0;ecy=0.0
if axis==1:
    ecy=eccXY

if axis==2:
    ecx=eccXY


RFid1 = a.ReferencePoint(point=(-1.0*ecx, -1.0*ecy, 0.0)).id
RFid2 = a.ReferencePoint(point=(-1.0*ecx, -1.0*ecy, L)).id
r1 = a.referencePoints
refPoints1=(r1[RFid1], );refPoints2=(r1[RFid2], )

mdb.models[mod1].StaticStep(name='Step-1', previous='Initial',maxNumInc=10000, initialInc=0.25)
a.Set(referencePoints=refPoints2, name='Set-3')
regionDef=mdb.models[mod1].rootAssembly.sets['Set-3']
mdb.models[mod1].HistoryOutputRequest(name='H-Output-2', createStepName='Step-1', variables=('RF3', ), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)

mdb.models[mod1].fieldOutputRequests['F-Output-1'].suppress()
mdb.models[mod1].historyOutputRequests['H-Output-1'].suppress()

a = mdb.models[mod1].rootAssembly
r1 = a.referencePoints
refPoints1=(r1[RFid1], )
region = regionToolset.Region(referencePoints=refPoints1)
mdb.models[mod1].DisplacementBC(name='BC-1', createStepName='Step-1',  region=region, u1=0.0, u2=0.0, u3=L/350.0*(1.6+lam*0.05+eccXY/H*10.0), ur1=UNSET, ur2=UNSET, ur3=0.0, amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='',  localCsys=None)
a = mdb.models[mod1].rootAssembly
r1 = a.referencePoints
refPoints2=(r1[RFid2], )
region = regionToolset.Region(referencePoints=refPoints2)
mdb.models[mod1].DisplacementBC(name='BC-2', createStepName='Step-1', region=region, u1=0.0, u2=0.0, u3=0.0, ur1=UNSET, ur2=UNSET, ur3=0.0, amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='',   localCsys=None)

a1 = mdb.models[mod1].rootAssembly
a1.regenerate()


a = mdb.models[mod1].rootAssembly
e1 = a.instances['Part-3-1'].edges
edges1 = e1.getSequenceFromMask(mask=('[#1 ]', ), )
e2 = a.instances['Part-3-1-lin-1-2'].edges
edges1 = edges1+e2.getSequenceFromMask(mask=('[#1 ]', ), )
e3 = a.instances['Part-3-1-lin-2-1'].edges
edges1 = edges1+e3.getSequenceFromMask(mask=('[#1 ]', ), )
e4 = a.instances['Part-3-1-lin-2-2'].edges
edges1 = edges1+e4.getSequenceFromMask(mask=('[#1 ]', ), )
for i in range(n_Rep):
    e5 = a.instances['Part-4-'+str(i+1)].edges
    edges1 = edges1+e5.getSequenceFromMask(mask=('[#f ]', ), )


region1=regionToolset.Region(edges=edges1)
mdb.models[mod1].EmbeddedRegion(name='Constraint-4', embeddedRegion=region1, 
    hostRegion=None, weightFactorTolerance=1e-06, absoluteTolerance=0.0, 
    fractionalTolerance=0.05, toleranceMethod=BOTH)














f1 = a.instances['Part-1-1'].faces.getByBoundingBox(-B2-100-1.0, -H2-100-1.0, -1.0,          B2+100+1.0, H2+100+1.0,0.01)
f2 = a.instances['Part-1-1'].faces.getByBoundingBox(-B2-100-1.0, -H2-100-1.0, L-0.1, B2+100+1.0, H2+100+1.0,L+0.1) 
f11 = a.Surface(side1Faces=f1, name='Surface_Set1')
f22 = a.Surface(side1Faces=f2, name='Surface_Set2')

mdb.models[mod1].Coupling(name='Constraint-2', controlPoint=refPoints1, surface=f11, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
mdb.models[mod1].Coupling(name='Constraint-3', controlPoint=refPoints2, surface=f22, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)


a1 = mdb.models[mod1].rootAssembly
a1.regenerate()
a = mdb.models[mod1].rootAssembly
mdb.Job(name=mod1, model=mod1, description='', type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
    memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,   explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
    modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',  scratch='', multiprocessingMode=DEFAULT, numCpus=1, numGPUs=0)
mdb.jobs[mod1].submit(consistencyChecking=OFF)

