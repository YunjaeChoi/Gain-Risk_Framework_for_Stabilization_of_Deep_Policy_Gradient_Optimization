ū
-č,
,
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
.
IsFinite
x"T
y
"
Ttype:
2
2
L2Loss
t"T
output"T"
Ttype:
2
\
	LeakyRelu
features"T
activations"T"
alphafloat%ĶĢL>"
Ttype0:
2
n
LeakyReluGrad
	gradients"T
features"T
	backprops"T"
alphafloat%ĶĢL>"
Ttype0:
2
:
Less
x"T
y"T
z
"
Ttype:
2	
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
ą
ResourceApplyAdam
var
m
v
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
/
Sign
x"T
y"T"
Ttype:

2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
@
Softplus
features"T
activations"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized

&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.14.02unknowną’
n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’
p
Placeholder_1Placeholder*
shape:’’’’’’’’’*
dtype0*'
_output_shapes
:’’’’’’’’’
p
Placeholder_2Placeholder*
shape:’’’’’’’’’*
dtype0*'
_output_shapes
:’’’’’’’’’
h
Placeholder_3Placeholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’
h
Placeholder_4Placeholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’
h
Placeholder_5Placeholder*
shape:’’’’’’’’’*
dtype0*#
_output_shapes
:’’’’’’’’’
h
Placeholder_6Placeholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’
h
Placeholder_7Placeholder*
dtype0*#
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’
p
Placeholder_8Placeholder*
shape:’’’’’’’’’*
dtype0*'
_output_shapes
:’’’’’’’’’
p
Placeholder_9Placeholder*
dtype0*'
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’
½
<pi/actor/actor/dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *.
_class$
" loc:@pi/actor/actor/dense/kernel*
dtype0
Æ
:pi/actor/actor/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *Ł¾*.
_class$
" loc:@pi/actor/actor/dense/kernel*
dtype0*
_output_shapes
: 
Æ
:pi/actor/actor/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *Ł>*.
_class$
" loc:@pi/actor/actor/dense/kernel*
dtype0*
_output_shapes
: 

Dpi/actor/actor/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform<pi/actor/actor/dense/kernel/Initializer/random_uniform/shape*.
_class$
" loc:@pi/actor/actor/dense/kernel*
seed2*
dtype0*
_output_shapes
:	*

seed *
T0

:pi/actor/actor/dense/kernel/Initializer/random_uniform/subSub:pi/actor/actor/dense/kernel/Initializer/random_uniform/max:pi/actor/actor/dense/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@pi/actor/actor/dense/kernel*
_output_shapes
: 

:pi/actor/actor/dense/kernel/Initializer/random_uniform/mulMulDpi/actor/actor/dense/kernel/Initializer/random_uniform/RandomUniform:pi/actor/actor/dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	*
T0*.
_class$
" loc:@pi/actor/actor/dense/kernel

6pi/actor/actor/dense/kernel/Initializer/random_uniformAdd:pi/actor/actor/dense/kernel/Initializer/random_uniform/mul:pi/actor/actor/dense/kernel/Initializer/random_uniform/min*
T0*.
_class$
" loc:@pi/actor/actor/dense/kernel*
_output_shapes
:	
Ō
pi/actor/actor/dense/kernelVarHandleOp*
shape:	*
dtype0*
_output_shapes
: *,
shared_namepi/actor/actor/dense/kernel*.
_class$
" loc:@pi/actor/actor/dense/kernel*
	container 

<pi/actor/actor/dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOppi/actor/actor/dense/kernel*
_output_shapes
: 
Č
"pi/actor/actor/dense/kernel/AssignAssignVariableOppi/actor/actor/dense/kernel6pi/actor/actor/dense/kernel/Initializer/random_uniform*
dtype0*.
_class$
" loc:@pi/actor/actor/dense/kernel
¼
/pi/actor/actor/dense/kernel/Read/ReadVariableOpReadVariableOppi/actor/actor/dense/kernel*.
_class$
" loc:@pi/actor/actor/dense/kernel*
dtype0*
_output_shapes
:	
Ø
+pi/actor/actor/dense/bias/Initializer/zerosConst*
valueB*    *,
_class"
 loc:@pi/actor/actor/dense/bias*
dtype0*
_output_shapes	
:
Ź
pi/actor/actor/dense/biasVarHandleOp*
dtype0*
_output_shapes
: **
shared_namepi/actor/actor/dense/bias*,
_class"
 loc:@pi/actor/actor/dense/bias*
	container *
shape:

:pi/actor/actor/dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOppi/actor/actor/dense/bias*
_output_shapes
: 
·
 pi/actor/actor/dense/bias/AssignAssignVariableOppi/actor/actor/dense/bias+pi/actor/actor/dense/bias/Initializer/zeros*,
_class"
 loc:@pi/actor/actor/dense/bias*
dtype0
²
-pi/actor/actor/dense/bias/Read/ReadVariableOpReadVariableOppi/actor/actor/dense/bias*,
_class"
 loc:@pi/actor/actor/dense/bias*
dtype0*
_output_shapes	
:

*pi/actor/actor/dense/MatMul/ReadVariableOpReadVariableOppi/actor/actor/dense/kernel*
dtype0*
_output_shapes
:	
·
pi/actor/actor/dense/MatMulMatMulPlaceholder*pi/actor/actor/dense/MatMul/ReadVariableOp*
transpose_b( *
T0*(
_output_shapes
:’’’’’’’’’*
transpose_a( 

+pi/actor/actor/dense/BiasAdd/ReadVariableOpReadVariableOppi/actor/actor/dense/bias*
dtype0*
_output_shapes	
:
»
pi/actor/actor/dense/BiasAddBiasAddpi/actor/actor/dense/MatMul+pi/actor/actor/dense/BiasAdd/ReadVariableOp*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’*
T0

$pi/actor/actor/leaky_re_lu/LeakyRelu	LeakyRelupi/actor/actor/dense/BiasAdd*
T0*
alpha%>*(
_output_shapes
:’’’’’’’’’
Į
>pi/actor/actor/dense_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *0
_class&
$"loc:@pi/actor/actor/dense_1/kernel
³
<pi/actor/actor/dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *   ¾*0
_class&
$"loc:@pi/actor/actor/dense_1/kernel
³
<pi/actor/actor/dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *   >*0
_class&
$"loc:@pi/actor/actor/dense_1/kernel

Fpi/actor/actor/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform>pi/actor/actor/dense_1/kernel/Initializer/random_uniform/shape*0
_class&
$"loc:@pi/actor/actor/dense_1/kernel*
seed2"*
dtype0* 
_output_shapes
:
*

seed *
T0

<pi/actor/actor/dense_1/kernel/Initializer/random_uniform/subSub<pi/actor/actor/dense_1/kernel/Initializer/random_uniform/max<pi/actor/actor/dense_1/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@pi/actor/actor/dense_1/kernel*
_output_shapes
: 
¦
<pi/actor/actor/dense_1/kernel/Initializer/random_uniform/mulMulFpi/actor/actor/dense_1/kernel/Initializer/random_uniform/RandomUniform<pi/actor/actor/dense_1/kernel/Initializer/random_uniform/sub*
T0*0
_class&
$"loc:@pi/actor/actor/dense_1/kernel* 
_output_shapes
:


8pi/actor/actor/dense_1/kernel/Initializer/random_uniformAdd<pi/actor/actor/dense_1/kernel/Initializer/random_uniform/mul<pi/actor/actor/dense_1/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@pi/actor/actor/dense_1/kernel* 
_output_shapes
:

Ū
pi/actor/actor/dense_1/kernelVarHandleOp*
	container *
shape:
*
dtype0*
_output_shapes
: *.
shared_namepi/actor/actor/dense_1/kernel*0
_class&
$"loc:@pi/actor/actor/dense_1/kernel

>pi/actor/actor/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOppi/actor/actor/dense_1/kernel*
_output_shapes
: 
Š
$pi/actor/actor/dense_1/kernel/AssignAssignVariableOppi/actor/actor/dense_1/kernel8pi/actor/actor/dense_1/kernel/Initializer/random_uniform*0
_class&
$"loc:@pi/actor/actor/dense_1/kernel*
dtype0
Ć
1pi/actor/actor/dense_1/kernel/Read/ReadVariableOpReadVariableOppi/actor/actor/dense_1/kernel*0
_class&
$"loc:@pi/actor/actor/dense_1/kernel*
dtype0* 
_output_shapes
:

¬
-pi/actor/actor/dense_1/bias/Initializer/zerosConst*
valueB*    *.
_class$
" loc:@pi/actor/actor/dense_1/bias*
dtype0*
_output_shapes	
:
Š
pi/actor/actor/dense_1/biasVarHandleOp*,
shared_namepi/actor/actor/dense_1/bias*.
_class$
" loc:@pi/actor/actor/dense_1/bias*
	container *
shape:*
dtype0*
_output_shapes
: 

<pi/actor/actor/dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOppi/actor/actor/dense_1/bias*
_output_shapes
: 
æ
"pi/actor/actor/dense_1/bias/AssignAssignVariableOppi/actor/actor/dense_1/bias-pi/actor/actor/dense_1/bias/Initializer/zeros*.
_class$
" loc:@pi/actor/actor/dense_1/bias*
dtype0
ø
/pi/actor/actor/dense_1/bias/Read/ReadVariableOpReadVariableOppi/actor/actor/dense_1/bias*
_output_shapes	
:*.
_class$
" loc:@pi/actor/actor/dense_1/bias*
dtype0

,pi/actor/actor/dense_1/MatMul/ReadVariableOpReadVariableOppi/actor/actor/dense_1/kernel*
dtype0* 
_output_shapes
:

Ō
pi/actor/actor/dense_1/MatMulMatMul$pi/actor/actor/leaky_re_lu/LeakyRelu,pi/actor/actor/dense_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b( 

-pi/actor/actor/dense_1/BiasAdd/ReadVariableOpReadVariableOppi/actor/actor/dense_1/bias*
_output_shapes	
:*
dtype0
Į
pi/actor/actor/dense_1/BiasAddBiasAddpi/actor/actor/dense_1/MatMul-pi/actor/actor/dense_1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’

&pi/actor/actor/leaky_re_lu_1/LeakyRelu	LeakyRelupi/actor/actor/dense_1/BiasAdd*
T0*
alpha%>*(
_output_shapes
:’’’’’’’’’
Į
>pi/actor/actor/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *0
_class&
$"loc:@pi/actor/actor/dense_2/kernel*
dtype0*
_output_shapes
:
³
<pi/actor/actor/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *¦D»*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel*
dtype0*
_output_shapes
: 
³
<pi/actor/actor/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *¦D;*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel*
dtype0*
_output_shapes
: 

Fpi/actor/actor/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform>pi/actor/actor/dense_2/kernel/Initializer/random_uniform/shape*
T0*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel*
seed27*
dtype0*
_output_shapes
:	*

seed 

<pi/actor/actor/dense_2/kernel/Initializer/random_uniform/subSub<pi/actor/actor/dense_2/kernel/Initializer/random_uniform/max<pi/actor/actor/dense_2/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel*
_output_shapes
: 
„
<pi/actor/actor/dense_2/kernel/Initializer/random_uniform/mulMulFpi/actor/actor/dense_2/kernel/Initializer/random_uniform/RandomUniform<pi/actor/actor/dense_2/kernel/Initializer/random_uniform/sub*
T0*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel*
_output_shapes
:	

8pi/actor/actor/dense_2/kernel/Initializer/random_uniformAdd<pi/actor/actor/dense_2/kernel/Initializer/random_uniform/mul<pi/actor/actor/dense_2/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel*
_output_shapes
:	
Ś
pi/actor/actor/dense_2/kernelVarHandleOp*.
shared_namepi/actor/actor/dense_2/kernel*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel*
	container *
shape:	*
dtype0*
_output_shapes
: 

>pi/actor/actor/dense_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOppi/actor/actor/dense_2/kernel*
_output_shapes
: 
Š
$pi/actor/actor/dense_2/kernel/AssignAssignVariableOppi/actor/actor/dense_2/kernel8pi/actor/actor/dense_2/kernel/Initializer/random_uniform*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel*
dtype0
Ā
1pi/actor/actor/dense_2/kernel/Read/ReadVariableOpReadVariableOppi/actor/actor/dense_2/kernel*
dtype0*
_output_shapes
:	*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel
Ŗ
-pi/actor/actor/dense_2/bias/Initializer/zerosConst*
valueB*    *.
_class$
" loc:@pi/actor/actor/dense_2/bias*
dtype0*
_output_shapes
:
Ļ
pi/actor/actor/dense_2/biasVarHandleOp*
dtype0*
_output_shapes
: *,
shared_namepi/actor/actor/dense_2/bias*.
_class$
" loc:@pi/actor/actor/dense_2/bias*
	container *
shape:

<pi/actor/actor/dense_2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOppi/actor/actor/dense_2/bias*
_output_shapes
: 
æ
"pi/actor/actor/dense_2/bias/AssignAssignVariableOppi/actor/actor/dense_2/bias-pi/actor/actor/dense_2/bias/Initializer/zeros*.
_class$
" loc:@pi/actor/actor/dense_2/bias*
dtype0
·
/pi/actor/actor/dense_2/bias/Read/ReadVariableOpReadVariableOppi/actor/actor/dense_2/bias*
dtype0*
_output_shapes
:*.
_class$
" loc:@pi/actor/actor/dense_2/bias

,pi/actor/actor/dense_2/MatMul/ReadVariableOpReadVariableOppi/actor/actor/dense_2/kernel*
dtype0*
_output_shapes
:	
Õ
pi/actor/actor/dense_2/MatMulMatMul&pi/actor/actor/leaky_re_lu_1/LeakyRelu,pi/actor/actor/dense_2/MatMul/ReadVariableOp*'
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b( *
T0

-pi/actor/actor/dense_2/BiasAdd/ReadVariableOpReadVariableOppi/actor/actor/dense_2/bias*
dtype0*
_output_shapes
:
Ą
pi/actor/actor/dense_2/BiasAddBiasAddpi/actor/actor/dense_2/MatMul-pi/actor/actor/dense_2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’
Į
>pi/actor/actor/dense_3/kernel/Initializer/random_uniform/shapeConst*
valueB"      *0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*
dtype0*
_output_shapes
:
³
<pi/actor/actor/dense_3/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *¦D»*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*
dtype0
³
<pi/actor/actor/dense_3/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *¦D;*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*
dtype0

Fpi/actor/actor/dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform>pi/actor/actor/dense_3/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed *
T0*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*
seed2K

<pi/actor/actor/dense_3/kernel/Initializer/random_uniform/subSub<pi/actor/actor/dense_3/kernel/Initializer/random_uniform/max<pi/actor/actor/dense_3/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*
_output_shapes
: 
„
<pi/actor/actor/dense_3/kernel/Initializer/random_uniform/mulMulFpi/actor/actor/dense_3/kernel/Initializer/random_uniform/RandomUniform<pi/actor/actor/dense_3/kernel/Initializer/random_uniform/sub*
T0*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*
_output_shapes
:	

8pi/actor/actor/dense_3/kernel/Initializer/random_uniformAdd<pi/actor/actor/dense_3/kernel/Initializer/random_uniform/mul<pi/actor/actor/dense_3/kernel/Initializer/random_uniform/min*
T0*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*
_output_shapes
:	
Ś
pi/actor/actor/dense_3/kernelVarHandleOp*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*
	container *
shape:	*
dtype0*
_output_shapes
: *.
shared_namepi/actor/actor/dense_3/kernel

>pi/actor/actor/dense_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOppi/actor/actor/dense_3/kernel*
_output_shapes
: 
Š
$pi/actor/actor/dense_3/kernel/AssignAssignVariableOppi/actor/actor/dense_3/kernel8pi/actor/actor/dense_3/kernel/Initializer/random_uniform*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*
dtype0
Ā
1pi/actor/actor/dense_3/kernel/Read/ReadVariableOpReadVariableOppi/actor/actor/dense_3/kernel*
_output_shapes
:	*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*
dtype0
Ŗ
-pi/actor/actor/dense_3/bias/Initializer/zerosConst*
_output_shapes
:*
valueB*    *.
_class$
" loc:@pi/actor/actor/dense_3/bias*
dtype0
Ļ
pi/actor/actor/dense_3/biasVarHandleOp*.
_class$
" loc:@pi/actor/actor/dense_3/bias*
	container *
shape:*
dtype0*
_output_shapes
: *,
shared_namepi/actor/actor/dense_3/bias

<pi/actor/actor/dense_3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOppi/actor/actor/dense_3/bias*
_output_shapes
: 
æ
"pi/actor/actor/dense_3/bias/AssignAssignVariableOppi/actor/actor/dense_3/bias-pi/actor/actor/dense_3/bias/Initializer/zeros*.
_class$
" loc:@pi/actor/actor/dense_3/bias*
dtype0
·
/pi/actor/actor/dense_3/bias/Read/ReadVariableOpReadVariableOppi/actor/actor/dense_3/bias*.
_class$
" loc:@pi/actor/actor/dense_3/bias*
dtype0*
_output_shapes
:

,pi/actor/actor/dense_3/MatMul/ReadVariableOpReadVariableOppi/actor/actor/dense_3/kernel*
dtype0*
_output_shapes
:	
Õ
pi/actor/actor/dense_3/MatMulMatMul&pi/actor/actor/leaky_re_lu_1/LeakyRelu,pi/actor/actor/dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b( 

-pi/actor/actor/dense_3/BiasAdd/ReadVariableOpReadVariableOppi/actor/actor/dense_3/bias*
dtype0*
_output_shapes
:
Ą
pi/actor/actor/dense_3/BiasAddBiasAddpi/actor/actor/dense_3/MatMul-pi/actor/actor/dense_3/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’
}
pi/actor/actor/dense_3/SoftplusSoftpluspi/actor/actor/dense_3/BiasAdd*'
_output_shapes
:’’’’’’’’’*
T0
p
pi/Normal/IdentityIdentitypi/actor/actor/dense_2/BiasAdd*
T0*'
_output_shapes
:’’’’’’’’’
s
pi/Normal/Identity_1Identitypi/actor/actor/dense_3/Softplus*'
_output_shapes
:’’’’’’’’’*
T0
`
pi/Normal/sample/sample_shapeConst*
valueB *
dtype0*
_output_shapes
: 
t
"pi/Normal/batch_shape_tensor/ShapeShapepi/Normal/Identity*
T0*
out_type0*
_output_shapes
:
x
$pi/Normal/batch_shape_tensor/Shape_1Shapepi/Normal/Identity_1*
T0*
out_type0*
_output_shapes
:
Ŗ
*pi/Normal/batch_shape_tensor/BroadcastArgsBroadcastArgs"pi/Normal/batch_shape_tensor/Shape$pi/Normal/batch_shape_tensor/Shape_1*
T0*
_output_shapes
:
j
 pi/Normal/sample/concat/values_0Const*
valueB:*
dtype0*
_output_shapes
:
^
pi/Normal/sample/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
É
pi/Normal/sample/concatConcatV2 pi/Normal/sample/concat/values_0*pi/Normal/batch_shape_tensor/BroadcastArgspi/Normal/sample/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
h
#pi/Normal/sample/random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
j
%pi/Normal/sample/random_normal/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
É
3pi/Normal/sample/random_normal/RandomStandardNormalRandomStandardNormalpi/Normal/sample/concat*
T0*
dtype0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’*
seed2h*

seed 
Ä
"pi/Normal/sample/random_normal/mulMul3pi/Normal/sample/random_normal/RandomStandardNormal%pi/Normal/sample/random_normal/stddev*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’*
T0
­
pi/Normal/sample/random_normalAdd"pi/Normal/sample/random_normal/mul#pi/Normal/sample/random_normal/mean*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’

pi/Normal/sample/mulMulpi/Normal/sample/random_normalpi/Normal/Identity_1*
T0*+
_output_shapes
:’’’’’’’’’
{
pi/Normal/sample/addAddpi/Normal/sample/mulpi/Normal/Identity*
T0*+
_output_shapes
:’’’’’’’’’
j
pi/Normal/sample/ShapeShapepi/Normal/sample/add*
T0*
out_type0*
_output_shapes
:
n
$pi/Normal/sample/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
p
&pi/Normal/sample/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
p
&pi/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Ņ
pi/Normal/sample/strided_sliceStridedSlicepi/Normal/sample/Shape$pi/Normal/sample/strided_slice/stack&pi/Normal/sample/strided_slice/stack_1&pi/Normal/sample/strided_slice/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
Index0*
T0
`
pi/Normal/sample/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
¾
pi/Normal/sample/concat_1ConcatV2pi/Normal/sample/sample_shapepi/Normal/sample/strided_slicepi/Normal/sample/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:

pi/Normal/sample/ReshapeReshapepi/Normal/sample/addpi/Normal/sample/concat_1*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’

"pi/Normal/log_prob/standardize/subSubpi/Normal/sample/Reshapepi/Normal/Identity*'
_output_shapes
:’’’’’’’’’*
T0

&pi/Normal/log_prob/standardize/truedivRealDiv"pi/Normal/log_prob/standardize/subpi/Normal/Identity_1*
T0*'
_output_shapes
:’’’’’’’’’
}
pi/Normal/log_prob/SquareSquare&pi/Normal/log_prob/standardize/truediv*
T0*'
_output_shapes
:’’’’’’’’’
]
pi/Normal/log_prob/mul/xConst*
valueB
 *   æ*
dtype0*
_output_shapes
: 

pi/Normal/log_prob/mulMulpi/Normal/log_prob/mul/xpi/Normal/log_prob/Square*
T0*'
_output_shapes
:’’’’’’’’’
e
pi/Normal/log_prob/LogLogpi/Normal/Identity_1*
T0*'
_output_shapes
:’’’’’’’’’
]
pi/Normal/log_prob/add/xConst*
valueB
 *?k?*
dtype0*
_output_shapes
: 

pi/Normal/log_prob/addAddpi/Normal/log_prob/add/xpi/Normal/log_prob/Log*'
_output_shapes
:’’’’’’’’’*
T0

pi/Normal/log_prob/subSubpi/Normal/log_prob/mulpi/Normal/log_prob/add*'
_output_shapes
:’’’’’’’’’*
T0
Z
pi/Sum/reduction_indicesConst*
_output_shapes
: *
value	B :*
dtype0

pi/SumSumpi/Normal/log_prob/subpi/Sum/reduction_indices*#
_output_shapes
:’’’’’’’’’*
	keep_dims( *

Tidx0*
T0
M
pi/sub/xConst*
valueB
 *r1?*
dtype0*
_output_shapes
: 
c
pi/subSubpi/sub/xpi/Normal/sample/Reshape*
T0*'
_output_shapes
:’’’’’’’’’
M
pi/mul/xConst*
valueB
 *   Ą*
dtype0*
_output_shapes
: 
c
pi/mulMulpi/mul/xpi/Normal/sample/Reshape*'
_output_shapes
:’’’’’’’’’*
T0
Q
pi/SoftplusSoftpluspi/mul*'
_output_shapes
:’’’’’’’’’*
T0
V
pi/sub_1Subpi/subpi/Softplus*
T0*'
_output_shapes
:’’’’’’’’’
O

pi/mul_1/xConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
W
pi/mul_1Mul
pi/mul_1/xpi/sub_1*'
_output_shapes
:’’’’’’’’’*
T0
\
pi/Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 

pi/Sum_1Sumpi/mul_1pi/Sum_1/reduction_indices*
T0*#
_output_shapes
:’’’’’’’’’*
	keep_dims( *

Tidx0
O
pi/sub_2Subpi/Sumpi/Sum_1*
T0*#
_output_shapes
:’’’’’’’’’
[
pi/TanhTanhpi/Normal/sample/Reshape*
T0*'
_output_shapes
:’’’’’’’’’

$pi/Normal/log_prob_1/standardize/subSubPlaceholder_1pi/Normal/Identity*'
_output_shapes
:’’’’’’’’’*
T0
”
(pi/Normal/log_prob_1/standardize/truedivRealDiv$pi/Normal/log_prob_1/standardize/subpi/Normal/Identity_1*
T0*'
_output_shapes
:’’’’’’’’’

pi/Normal/log_prob_1/SquareSquare(pi/Normal/log_prob_1/standardize/truediv*'
_output_shapes
:’’’’’’’’’*
T0
_
pi/Normal/log_prob_1/mul/xConst*
valueB
 *   æ*
dtype0*
_output_shapes
: 

pi/Normal/log_prob_1/mulMulpi/Normal/log_prob_1/mul/xpi/Normal/log_prob_1/Square*
T0*'
_output_shapes
:’’’’’’’’’
g
pi/Normal/log_prob_1/LogLogpi/Normal/Identity_1*'
_output_shapes
:’’’’’’’’’*
T0
_
pi/Normal/log_prob_1/add/xConst*
valueB
 *?k?*
dtype0*
_output_shapes
: 

pi/Normal/log_prob_1/addAddpi/Normal/log_prob_1/add/xpi/Normal/log_prob_1/Log*
T0*'
_output_shapes
:’’’’’’’’’

pi/Normal/log_prob_1/subSubpi/Normal/log_prob_1/mulpi/Normal/log_prob_1/add*
T0*'
_output_shapes
:’’’’’’’’’
\
pi/Sum_2/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 

pi/Sum_2Sumpi/Normal/log_prob_1/subpi/Sum_2/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:’’’’’’’’’
O

pi/sub_3/xConst*
valueB
 *r1?*
dtype0*
_output_shapes
: 
\
pi/sub_3Sub
pi/sub_3/xPlaceholder_1*'
_output_shapes
:’’’’’’’’’*
T0
O

pi/mul_2/xConst*
valueB
 *   Ą*
dtype0*
_output_shapes
: 
\
pi/mul_2Mul
pi/mul_2/xPlaceholder_1*'
_output_shapes
:’’’’’’’’’*
T0
U
pi/Softplus_1Softpluspi/mul_2*'
_output_shapes
:’’’’’’’’’*
T0
Z
pi/sub_4Subpi/sub_3pi/Softplus_1*'
_output_shapes
:’’’’’’’’’*
T0
O

pi/mul_3/xConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
W
pi/mul_3Mul
pi/mul_3/xpi/sub_4*
T0*'
_output_shapes
:’’’’’’’’’
\
pi/Sum_3/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :

pi/Sum_3Sumpi/mul_3pi/Sum_3/reduction_indices*#
_output_shapes
:’’’’’’’’’*
	keep_dims( *

Tidx0*
T0
Q
pi/sub_5Subpi/Sum_2pi/Sum_3*#
_output_shapes
:’’’’’’’’’*
T0
R
	pi/Tanh_1TanhPlaceholder_1*
T0*'
_output_shapes
:’’’’’’’’’
Ć
?v/critic/critic/dense_4/kernel/Initializer/random_uniform/shapeConst*
valueB"      *1
_class'
%#loc:@v/critic/critic/dense_4/kernel*
dtype0*
_output_shapes
:
µ
=v/critic/critic/dense_4/kernel/Initializer/random_uniform/minConst*
valueB
 *Ł¾*1
_class'
%#loc:@v/critic/critic/dense_4/kernel*
dtype0*
_output_shapes
: 
µ
=v/critic/critic/dense_4/kernel/Initializer/random_uniform/maxConst*
valueB
 *Ł>*1
_class'
%#loc:@v/critic/critic/dense_4/kernel*
dtype0*
_output_shapes
: 

Gv/critic/critic/dense_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform?v/critic/critic/dense_4/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed *
T0*1
_class'
%#loc:@v/critic/critic/dense_4/kernel*
seed2¦

=v/critic/critic/dense_4/kernel/Initializer/random_uniform/subSub=v/critic/critic/dense_4/kernel/Initializer/random_uniform/max=v/critic/critic/dense_4/kernel/Initializer/random_uniform/min*
T0*1
_class'
%#loc:@v/critic/critic/dense_4/kernel*
_output_shapes
: 
©
=v/critic/critic/dense_4/kernel/Initializer/random_uniform/mulMulGv/critic/critic/dense_4/kernel/Initializer/random_uniform/RandomUniform=v/critic/critic/dense_4/kernel/Initializer/random_uniform/sub*
T0*1
_class'
%#loc:@v/critic/critic/dense_4/kernel*
_output_shapes
:	

9v/critic/critic/dense_4/kernel/Initializer/random_uniformAdd=v/critic/critic/dense_4/kernel/Initializer/random_uniform/mul=v/critic/critic/dense_4/kernel/Initializer/random_uniform/min*
_output_shapes
:	*
T0*1
_class'
%#loc:@v/critic/critic/dense_4/kernel
Ż
v/critic/critic/dense_4/kernelVarHandleOp*/
shared_name v/critic/critic/dense_4/kernel*1
_class'
%#loc:@v/critic/critic/dense_4/kernel*
	container *
shape:	*
dtype0*
_output_shapes
: 

?v/critic/critic/dense_4/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpv/critic/critic/dense_4/kernel*
_output_shapes
: 
Ō
%v/critic/critic/dense_4/kernel/AssignAssignVariableOpv/critic/critic/dense_4/kernel9v/critic/critic/dense_4/kernel/Initializer/random_uniform*1
_class'
%#loc:@v/critic/critic/dense_4/kernel*
dtype0
Å
2v/critic/critic/dense_4/kernel/Read/ReadVariableOpReadVariableOpv/critic/critic/dense_4/kernel*1
_class'
%#loc:@v/critic/critic/dense_4/kernel*
dtype0*
_output_shapes
:	
®
.v/critic/critic/dense_4/bias/Initializer/zerosConst*
valueB*    */
_class%
#!loc:@v/critic/critic/dense_4/bias*
dtype0*
_output_shapes	
:
Ó
v/critic/critic/dense_4/biasVarHandleOp*/
_class%
#!loc:@v/critic/critic/dense_4/bias*
	container *
shape:*
dtype0*
_output_shapes
: *-
shared_namev/critic/critic/dense_4/bias

=v/critic/critic/dense_4/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpv/critic/critic/dense_4/bias*
_output_shapes
: 
Ć
#v/critic/critic/dense_4/bias/AssignAssignVariableOpv/critic/critic/dense_4/bias.v/critic/critic/dense_4/bias/Initializer/zeros*/
_class%
#!loc:@v/critic/critic/dense_4/bias*
dtype0
»
0v/critic/critic/dense_4/bias/Read/ReadVariableOpReadVariableOpv/critic/critic/dense_4/bias*
_output_shapes	
:*/
_class%
#!loc:@v/critic/critic/dense_4/bias*
dtype0

-v/critic/critic/dense_4/MatMul/ReadVariableOpReadVariableOpv/critic/critic/dense_4/kernel*
dtype0*
_output_shapes
:	
½
v/critic/critic/dense_4/MatMulMatMulPlaceholder-v/critic/critic/dense_4/MatMul/ReadVariableOp*(
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b( *
T0

.v/critic/critic/dense_4/BiasAdd/ReadVariableOpReadVariableOpv/critic/critic/dense_4/bias*
dtype0*
_output_shapes	
:
Ä
v/critic/critic/dense_4/BiasAddBiasAddv/critic/critic/dense_4/MatMul.v/critic/critic/dense_4/BiasAdd/ReadVariableOp*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’*
T0

'v/critic/critic/leaky_re_lu_2/LeakyRelu	LeakyReluv/critic/critic/dense_4/BiasAdd*
T0*
alpha%>*(
_output_shapes
:’’’’’’’’’
Ć
?v/critic/critic/dense_5/kernel/Initializer/random_uniform/shapeConst*
valueB"      *1
_class'
%#loc:@v/critic/critic/dense_5/kernel*
dtype0*
_output_shapes
:
µ
=v/critic/critic/dense_5/kernel/Initializer/random_uniform/minConst*
valueB
 *   ¾*1
_class'
%#loc:@v/critic/critic/dense_5/kernel*
dtype0*
_output_shapes
: 
µ
=v/critic/critic/dense_5/kernel/Initializer/random_uniform/maxConst*
valueB
 *   >*1
_class'
%#loc:@v/critic/critic/dense_5/kernel*
dtype0*
_output_shapes
: 

Gv/critic/critic/dense_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform?v/critic/critic/dense_5/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*1
_class'
%#loc:@v/critic/critic/dense_5/kernel*
seed2»

=v/critic/critic/dense_5/kernel/Initializer/random_uniform/subSub=v/critic/critic/dense_5/kernel/Initializer/random_uniform/max=v/critic/critic/dense_5/kernel/Initializer/random_uniform/min*1
_class'
%#loc:@v/critic/critic/dense_5/kernel*
_output_shapes
: *
T0
Ŗ
=v/critic/critic/dense_5/kernel/Initializer/random_uniform/mulMulGv/critic/critic/dense_5/kernel/Initializer/random_uniform/RandomUniform=v/critic/critic/dense_5/kernel/Initializer/random_uniform/sub*
T0*1
_class'
%#loc:@v/critic/critic/dense_5/kernel* 
_output_shapes
:


9v/critic/critic/dense_5/kernel/Initializer/random_uniformAdd=v/critic/critic/dense_5/kernel/Initializer/random_uniform/mul=v/critic/critic/dense_5/kernel/Initializer/random_uniform/min*
T0*1
_class'
%#loc:@v/critic/critic/dense_5/kernel* 
_output_shapes
:

Ž
v/critic/critic/dense_5/kernelVarHandleOp*/
shared_name v/critic/critic/dense_5/kernel*1
_class'
%#loc:@v/critic/critic/dense_5/kernel*
	container *
shape:
*
dtype0*
_output_shapes
: 

?v/critic/critic/dense_5/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpv/critic/critic/dense_5/kernel*
_output_shapes
: 
Ō
%v/critic/critic/dense_5/kernel/AssignAssignVariableOpv/critic/critic/dense_5/kernel9v/critic/critic/dense_5/kernel/Initializer/random_uniform*1
_class'
%#loc:@v/critic/critic/dense_5/kernel*
dtype0
Ę
2v/critic/critic/dense_5/kernel/Read/ReadVariableOpReadVariableOpv/critic/critic/dense_5/kernel*1
_class'
%#loc:@v/critic/critic/dense_5/kernel*
dtype0* 
_output_shapes
:

®
.v/critic/critic/dense_5/bias/Initializer/zerosConst*
valueB*    */
_class%
#!loc:@v/critic/critic/dense_5/bias*
dtype0*
_output_shapes	
:
Ó
v/critic/critic/dense_5/biasVarHandleOp*
shape:*
dtype0*
_output_shapes
: *-
shared_namev/critic/critic/dense_5/bias*/
_class%
#!loc:@v/critic/critic/dense_5/bias*
	container 

=v/critic/critic/dense_5/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpv/critic/critic/dense_5/bias*
_output_shapes
: 
Ć
#v/critic/critic/dense_5/bias/AssignAssignVariableOpv/critic/critic/dense_5/bias.v/critic/critic/dense_5/bias/Initializer/zeros*/
_class%
#!loc:@v/critic/critic/dense_5/bias*
dtype0
»
0v/critic/critic/dense_5/bias/Read/ReadVariableOpReadVariableOpv/critic/critic/dense_5/bias*
dtype0*
_output_shapes	
:*/
_class%
#!loc:@v/critic/critic/dense_5/bias

-v/critic/critic/dense_5/MatMul/ReadVariableOpReadVariableOpv/critic/critic/dense_5/kernel*
dtype0* 
_output_shapes
:

Ł
v/critic/critic/dense_5/MatMulMatMul'v/critic/critic/leaky_re_lu_2/LeakyRelu-v/critic/critic/dense_5/MatMul/ReadVariableOp*(
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b( *
T0

.v/critic/critic/dense_5/BiasAdd/ReadVariableOpReadVariableOpv/critic/critic/dense_5/bias*
dtype0*
_output_shapes	
:
Ä
v/critic/critic/dense_5/BiasAddBiasAddv/critic/critic/dense_5/MatMul.v/critic/critic/dense_5/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’

'v/critic/critic/leaky_re_lu_3/LeakyRelu	LeakyReluv/critic/critic/dense_5/BiasAdd*
T0*
alpha%>*(
_output_shapes
:’’’’’’’’’
Ć
?v/critic/critic/dense_6/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *1
_class'
%#loc:@v/critic/critic/dense_6/kernel*
dtype0
µ
=v/critic/critic/dense_6/kernel/Initializer/random_uniform/minConst*
valueB
 *¦D»*1
_class'
%#loc:@v/critic/critic/dense_6/kernel*
dtype0*
_output_shapes
: 
µ
=v/critic/critic/dense_6/kernel/Initializer/random_uniform/maxConst*
valueB
 *¦D;*1
_class'
%#loc:@v/critic/critic/dense_6/kernel*
dtype0*
_output_shapes
: 

Gv/critic/critic/dense_6/kernel/Initializer/random_uniform/RandomUniformRandomUniform?v/critic/critic/dense_6/kernel/Initializer/random_uniform/shape*

seed *
T0*1
_class'
%#loc:@v/critic/critic/dense_6/kernel*
seed2Š*
dtype0*
_output_shapes
:	

=v/critic/critic/dense_6/kernel/Initializer/random_uniform/subSub=v/critic/critic/dense_6/kernel/Initializer/random_uniform/max=v/critic/critic/dense_6/kernel/Initializer/random_uniform/min*
T0*1
_class'
%#loc:@v/critic/critic/dense_6/kernel*
_output_shapes
: 
©
=v/critic/critic/dense_6/kernel/Initializer/random_uniform/mulMulGv/critic/critic/dense_6/kernel/Initializer/random_uniform/RandomUniform=v/critic/critic/dense_6/kernel/Initializer/random_uniform/sub*
T0*1
_class'
%#loc:@v/critic/critic/dense_6/kernel*
_output_shapes
:	

9v/critic/critic/dense_6/kernel/Initializer/random_uniformAdd=v/critic/critic/dense_6/kernel/Initializer/random_uniform/mul=v/critic/critic/dense_6/kernel/Initializer/random_uniform/min*1
_class'
%#loc:@v/critic/critic/dense_6/kernel*
_output_shapes
:	*
T0
Ż
v/critic/critic/dense_6/kernelVarHandleOp*
_output_shapes
: */
shared_name v/critic/critic/dense_6/kernel*1
_class'
%#loc:@v/critic/critic/dense_6/kernel*
	container *
shape:	*
dtype0

?v/critic/critic/dense_6/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpv/critic/critic/dense_6/kernel*
_output_shapes
: 
Ō
%v/critic/critic/dense_6/kernel/AssignAssignVariableOpv/critic/critic/dense_6/kernel9v/critic/critic/dense_6/kernel/Initializer/random_uniform*1
_class'
%#loc:@v/critic/critic/dense_6/kernel*
dtype0
Å
2v/critic/critic/dense_6/kernel/Read/ReadVariableOpReadVariableOpv/critic/critic/dense_6/kernel*1
_class'
%#loc:@v/critic/critic/dense_6/kernel*
dtype0*
_output_shapes
:	
¬
.v/critic/critic/dense_6/bias/Initializer/zerosConst*
valueB*    */
_class%
#!loc:@v/critic/critic/dense_6/bias*
dtype0*
_output_shapes
:
Ņ
v/critic/critic/dense_6/biasVarHandleOp*
dtype0*
_output_shapes
: *-
shared_namev/critic/critic/dense_6/bias*/
_class%
#!loc:@v/critic/critic/dense_6/bias*
	container *
shape:

=v/critic/critic/dense_6/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpv/critic/critic/dense_6/bias*
_output_shapes
: 
Ć
#v/critic/critic/dense_6/bias/AssignAssignVariableOpv/critic/critic/dense_6/bias.v/critic/critic/dense_6/bias/Initializer/zeros*/
_class%
#!loc:@v/critic/critic/dense_6/bias*
dtype0
ŗ
0v/critic/critic/dense_6/bias/Read/ReadVariableOpReadVariableOpv/critic/critic/dense_6/bias*/
_class%
#!loc:@v/critic/critic/dense_6/bias*
dtype0*
_output_shapes
:

-v/critic/critic/dense_6/MatMul/ReadVariableOpReadVariableOpv/critic/critic/dense_6/kernel*
dtype0*
_output_shapes
:	
Ų
v/critic/critic/dense_6/MatMulMatMul'v/critic/critic/leaky_re_lu_3/LeakyRelu-v/critic/critic/dense_6/MatMul/ReadVariableOp*'
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b( *
T0

.v/critic/critic/dense_6/BiasAdd/ReadVariableOpReadVariableOpv/critic/critic/dense_6/bias*
dtype0*
_output_shapes
:
Ć
v/critic/critic/dense_6/BiasAddBiasAddv/critic/critic/dense_6/MatMul.v/critic/critic/dense_6/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’
z
	v/SqueezeSqueezev/critic/critic/dense_6/BiasAdd*
T0*#
_output_shapes
:’’’’’’’’’*
squeeze_dims


/v/critic_1/critic/dense_4/MatMul/ReadVariableOpReadVariableOpv/critic/critic/dense_4/kernel*
dtype0*
_output_shapes
:	
Ć
 v/critic_1/critic/dense_4/MatMulMatMulPlaceholder_2/v/critic_1/critic/dense_4/MatMul/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b( 

0v/critic_1/critic/dense_4/BiasAdd/ReadVariableOpReadVariableOpv/critic/critic/dense_4/bias*
dtype0*
_output_shapes	
:
Ź
!v/critic_1/critic/dense_4/BiasAddBiasAdd v/critic_1/critic/dense_4/MatMul0v/critic_1/critic/dense_4/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’

)v/critic_1/critic/leaky_re_lu_2/LeakyRelu	LeakyRelu!v/critic_1/critic/dense_4/BiasAdd*(
_output_shapes
:’’’’’’’’’*
T0*
alpha%>

/v/critic_1/critic/dense_5/MatMul/ReadVariableOpReadVariableOpv/critic/critic/dense_5/kernel*
dtype0* 
_output_shapes
:

ß
 v/critic_1/critic/dense_5/MatMulMatMul)v/critic_1/critic/leaky_re_lu_2/LeakyRelu/v/critic_1/critic/dense_5/MatMul/ReadVariableOp*
transpose_b( *
T0*(
_output_shapes
:’’’’’’’’’*
transpose_a( 

0v/critic_1/critic/dense_5/BiasAdd/ReadVariableOpReadVariableOpv/critic/critic/dense_5/bias*
dtype0*
_output_shapes	
:
Ź
!v/critic_1/critic/dense_5/BiasAddBiasAdd v/critic_1/critic/dense_5/MatMul0v/critic_1/critic/dense_5/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*(
_output_shapes
:’’’’’’’’’

)v/critic_1/critic/leaky_re_lu_3/LeakyRelu	LeakyRelu!v/critic_1/critic/dense_5/BiasAdd*
alpha%>*(
_output_shapes
:’’’’’’’’’*
T0

/v/critic_1/critic/dense_6/MatMul/ReadVariableOpReadVariableOpv/critic/critic/dense_6/kernel*
dtype0*
_output_shapes
:	
Ž
 v/critic_1/critic/dense_6/MatMulMatMul)v/critic_1/critic/leaky_re_lu_3/LeakyRelu/v/critic_1/critic/dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b( 

0v/critic_1/critic/dense_6/BiasAdd/ReadVariableOpReadVariableOpv/critic/critic/dense_6/bias*
dtype0*
_output_shapes
:
É
!v/critic_1/critic/dense_6/BiasAddBiasAdd v/critic_1/critic/dense_6/MatMul0v/critic_1/critic/dense_6/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’
~
v/Squeeze_1Squeeze!v/critic_1/critic/dense_6/BiasAdd*#
_output_shapes
:’’’’’’’’’*
squeeze_dims
*
T0
Q
subSubpi/sub_5Placeholder_4*
T0*#
_output_shapes
:’’’’’’’’’
L
mulMulsubPlaceholder_3*#
_output_shapes
:’’’’’’’’’*
T0
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
V
MeanMeanmulConst*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
N
	Maximum/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
P
MaximumMaximummul	Maximum/y*
T0*#
_output_shapes
:’’’’’’’’’
N
	Minimum/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
P
MinimumMinimummul	Minimum/y*
T0*#
_output_shapes
:’’’’’’’’’
A
AbsAbsMinimum*
T0*#
_output_shapes
:’’’’’’’’’
?
Abs_1Abssub*
T0*#
_output_shapes
:’’’’’’’’’
L
mul_1/xConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
J
mul_1Mulmul_1/xAbs_1*
T0*#
_output_shapes
:’’’’’’’’’
G
SoftmaxSoftmaxmul_1*
T0*#
_output_shapes
:’’’’’’’’’
J
mul_2MulSoftmaxAbs_1*
T0*#
_output_shapes
:’’’’’’’’’
Q
Const_1Const*
valueB: *
dtype0*
_output_shapes
:
X
SumSummul_2Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
J
pow/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
7
powPowSumpow/y*
T0*
_output_shapes
: 
L
mul_3/xConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
H
mul_3Mulmul_3/xAbs*
T0*#
_output_shapes
:’’’’’’’’’
I
	Softmax_1Softmaxmul_3*
T0*#
_output_shapes
:’’’’’’’’’
N
mul_4Mul	Softmax_1Minimum*
T0*#
_output_shapes
:’’’’’’’’’
Q
Const_2Const*
valueB: *
dtype0*
_output_shapes
:
Z
Sum_1Summul_4Const_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
L
pow_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
=
pow_1PowSum_1pow_1/y*
T0*
_output_shapes
: 
L
mul_5/xConst*
valueB
 *ĶĢĢ=*
dtype0*
_output_shapes
: 
;
mul_5Mulmul_5/xpow*
T0*
_output_shapes
: 
L
mul_6/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
=
mul_6Mulmul_6/xpow_1*
T0*
_output_shapes
: 
9
addAddmul_5mul_6*
T0*
_output_shapes
: 
1
NegNegMean*
_output_shapes
: *
T0
7
add_1AddNegadd*
T0*
_output_shapes
: 
T
sub_1SubPlaceholder_6	v/Squeeze*
T0*#
_output_shapes
:’’’’’’’’’
L
pow_2/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
J
pow_2Powsub_1pow_2/y*
T0*#
_output_shapes
:’’’’’’’’’
Q
Const_3Const*
valueB: *
dtype0*
_output_shapes
:
\
Mean_1Meanpow_2Const_3*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
S
sub_2SubPlaceholder_4pi/sub_5*
T0*#
_output_shapes
:’’’’’’’’’
Q
Const_4Const*
valueB: *
dtype0*
_output_shapes
:
\
Mean_2Meansub_2Const_4*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
D
Neg_1Negpi/sub_5*#
_output_shapes
:’’’’’’’’’*
T0
Q
Const_5Const*
dtype0*
_output_shapes
:*
valueB: 
\
Mean_3MeanNeg_1Const_5*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
]
LogLogpi/actor/actor/dense_3/Softplus*'
_output_shapes
:’’’’’’’’’*
T0
Y
Sum_2/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
u
Sum_2SumLogSum_2/reduction_indices*#
_output_shapes
:’’’’’’’’’*
	keep_dims( *

Tidx0*
T0
L
Log_1/xConst*
_output_shapes
: *
valueB
 *Ą¢A*
dtype0
6
Log_1LogLog_1/x*
T0*
_output_shapes
: 
L
mul_7/xConst*
valueB
 *  @*
dtype0*
_output_shapes
: 
=
mul_7Mulmul_7/xLog_1*
T0*
_output_shapes
: 
H
add_2AddSum_2mul_7*#
_output_shapes
:’’’’’’’’’*
T0
Q
Const_6Const*
valueB: *
dtype0*
_output_shapes
:
\
Mean_4Meanadd_2Const_6*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
M
Log_2LogPlaceholder_9*'
_output_shapes
:’’’’’’’’’*
T0
Y
Sum_3/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
w
Sum_3SumLog_2Sum_3/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:’’’’’’’’’
L
Log_3/xConst*
valueB
 *Ą¢A*
dtype0*
_output_shapes
: 
6
Log_3LogLog_3/x*
_output_shapes
: *
T0
L
mul_8/xConst*
valueB
 *  @*
dtype0*
_output_shapes
: 
=
mul_8Mulmul_8/xLog_3*
_output_shapes
: *
T0
H
add_3AddSum_3mul_8*
T0*#
_output_shapes
:’’’’’’’’’
L
pow_3/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
V
pow_3PowPlaceholder_9pow_3/y*
T0*'
_output_shapes
:’’’’’’’’’
L
pow_4/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
h
pow_4Powpi/actor/actor/dense_3/Softpluspow_4/y*
T0*'
_output_shapes
:’’’’’’’’’
R
truedivRealDivpow_3pow_4*
T0*'
_output_shapes
:’’’’’’’’’
Y
Sum_4/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
y
Sum_4SumtruedivSum_4/reduction_indices*#
_output_shapes
:’’’’’’’’’*
	keep_dims( *

Tidx0*
T0
m
sub_3Subpi/actor/actor/dense_2/BiasAddPlaceholder_8*'
_output_shapes
:’’’’’’’’’*
T0
L
pow_5/yConst*
_output_shapes
: *
valueB
 *   @*
dtype0
N
pow_5Powsub_3pow_5/y*
T0*'
_output_shapes
:’’’’’’’’’
T
	truediv_1RealDivpow_5pow_4*'
_output_shapes
:’’’’’’’’’*
T0
Y
Sum_5/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
{
Sum_5Sum	truediv_1Sum_5/reduction_indices*#
_output_shapes
:’’’’’’’’’*
	keep_dims( *

Tidx0*
T0
T
	truediv_2RealDivpow_4pow_3*
T0*'
_output_shapes
:’’’’’’’’’
I
Log_4Log	truediv_2*'
_output_shapes
:’’’’’’’’’*
T0
Y
Sum_6/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
w
Sum_6SumLog_4Sum_6/reduction_indices*#
_output_shapes
:’’’’’’’’’*
	keep_dims( *

Tidx0*
T0
H
add_4AddSum_4Sum_5*
T0*#
_output_shapes
:’’’’’’’’’
H
add_5Addadd_4Sum_6*#
_output_shapes
:’’’’’’’’’*
T0
L
sub_4/yConst*
dtype0*
_output_shapes
: *
valueB
 *   A
J
sub_4Subadd_5sub_4/y*
T0*#
_output_shapes
:’’’’’’’’’
L
mul_9/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
J
mul_9Mulmul_9/xsub_4*#
_output_shapes
:’’’’’’’’’*
T0
Q
Const_7Const*
valueB: *
dtype0*
_output_shapes
:
\
Mean_5Meanmul_9Const_7*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
K
Less/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
G
LessLessmulLess/y*#
_output_shapes
:’’’’’’’’’*
T0
_
CastCastLess*

SrcT0
*
Truncate( *#
_output_shapes
:’’’’’’’’’*

DstT0
Q
Const_8Const*
valueB: *
dtype0*
_output_shapes
:
[
Mean_6MeanCastConst_8*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
Q
Const_9Const*
_output_shapes
:*
valueB: *
dtype0
\
Sum_7SumMaximumConst_9*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
R
Const_10Const*
valueB: *
dtype0*
_output_shapes
:
]
Sum_8SumMinimumConst_10*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
4
Abs_2AbsSum_8*
T0*
_output_shapes
: 
L
add_6/yConst*
valueB
 *wĢ+2*
dtype0*
_output_shapes
: 
=
add_6AddAbs_2add_6/y*
T0*
_output_shapes
: 
C
	truediv_3RealDivSum_7add_6*
_output_shapes
: *
T0
R
Const_11Const*
_output_shapes
:*
valueB: *
dtype0
]
Mean_7MeanAbs_1Const_11*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
R
Const_12Const*
valueB: *
dtype0*
_output_shapes
:
Y
MaxMaxAbs_1Const_12*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
R
Const_13Const*
valueB: *
dtype0*
_output_shapes
:
]
Max_1MaxMaximumConst_13*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
R
Const_14Const*
valueB: *
dtype0*
_output_shapes
:
[
MinMinMinimumConst_14*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
O
Placeholder_10Placeholder*
shape: *
dtype0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*

index_type0*
_output_shapes
: *
T0
>
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/Fill
µ
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/Fill&^gradients/add_1_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
·
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/Fill&^gradients/add_1_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
m
gradients/Neg_grad/NegNeg-gradients/add_1_grad/tuple/control_dependency*
T0*
_output_shapes
: 
]
#gradients/add_grad/tuple/group_depsNoOp0^gradients/add_1_grad/tuple/control_dependency_1
Ņ
+gradients/add_grad/tuple/control_dependencyIdentity/gradients/add_1_grad/tuple/control_dependency_1$^gradients/add_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
Ō
-gradients/add_grad/tuple/control_dependency_1Identity/gradients/add_1_grad/tuple/control_dependency_1$^gradients/add_grad/tuple/group_deps*
_output_shapes
: *
T0*!
_class
loc:@gradients/Fill
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Neg_grad/Neg!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
\
gradients/Mean_grad/ShapeShapemul*
T0*
out_type0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:’’’’’’’’’
^
gradients/Mean_grad/Shape_1Shapemul*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:’’’’’’’’’*
T0
r
gradients/mul_5_grad/MulMul+gradients/add_grad/tuple/control_dependencypow*
T0*
_output_shapes
: 
x
gradients/mul_5_grad/Mul_1Mul+gradients/add_grad/tuple/control_dependencymul_5/x*
T0*
_output_shapes
: 
e
%gradients/mul_5_grad/tuple/group_depsNoOp^gradients/mul_5_grad/Mul^gradients/mul_5_grad/Mul_1
É
-gradients/mul_5_grad/tuple/control_dependencyIdentitygradients/mul_5_grad/Mul&^gradients/mul_5_grad/tuple/group_deps*
_output_shapes
: *
T0*+
_class!
loc:@gradients/mul_5_grad/Mul
Ļ
/gradients/mul_5_grad/tuple/control_dependency_1Identitygradients/mul_5_grad/Mul_1&^gradients/mul_5_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_5_grad/Mul_1*
_output_shapes
: 
v
gradients/mul_6_grad/MulMul-gradients/add_grad/tuple/control_dependency_1pow_1*
_output_shapes
: *
T0
z
gradients/mul_6_grad/Mul_1Mul-gradients/add_grad/tuple/control_dependency_1mul_6/x*
T0*
_output_shapes
: 
e
%gradients/mul_6_grad/tuple/group_depsNoOp^gradients/mul_6_grad/Mul^gradients/mul_6_grad/Mul_1
É
-gradients/mul_6_grad/tuple/control_dependencyIdentitygradients/mul_6_grad/Mul&^gradients/mul_6_grad/tuple/group_deps*+
_class!
loc:@gradients/mul_6_grad/Mul*
_output_shapes
: *
T0
Ļ
/gradients/mul_6_grad/tuple/control_dependency_1Identitygradients/mul_6_grad/Mul_1&^gradients/mul_6_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_6_grad/Mul_1*
_output_shapes
: 
[
gradients/pow_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
]
gradients/pow_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
“
(gradients/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pow_grad/Shapegradients/pow_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
v
gradients/pow_grad/mulMul/gradients/mul_5_grad/tuple/control_dependency_1pow/y*
_output_shapes
: *
T0
]
gradients/pow_grad/sub/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
_
gradients/pow_grad/subSubpow/ygradients/pow_grad/sub/y*
T0*
_output_shapes
: 
[
gradients/pow_grad/PowPowSumgradients/pow_grad/sub*
_output_shapes
: *
T0
p
gradients/pow_grad/mul_1Mulgradients/pow_grad/mulgradients/pow_grad/Pow*
_output_shapes
: *
T0

gradients/pow_grad/SumSumgradients/pow_grad/mul_1(gradients/pow_grad/BroadcastGradientArgs*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

gradients/pow_grad/ReshapeReshapegradients/pow_grad/Sumgradients/pow_grad/Shape*
Tshape0*
_output_shapes
: *
T0
a
gradients/pow_grad/Greater/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
i
gradients/pow_grad/GreaterGreaterSumgradients/pow_grad/Greater/y*
T0*
_output_shapes
: 
e
"gradients/pow_grad/ones_like/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
g
"gradients/pow_grad/ones_like/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

gradients/pow_grad/ones_likeFill"gradients/pow_grad/ones_like/Shape"gradients/pow_grad/ones_like/Const*
_output_shapes
: *
T0*

index_type0

gradients/pow_grad/SelectSelectgradients/pow_grad/GreaterSumgradients/pow_grad/ones_like*
_output_shapes
: *
T0
Y
gradients/pow_grad/LogLoggradients/pow_grad/Select*
T0*
_output_shapes
: 
b
gradients/pow_grad/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/pow_grad/Select_1Selectgradients/pow_grad/Greatergradients/pow_grad/Loggradients/pow_grad/zeros_like*
T0*
_output_shapes
: 
v
gradients/pow_grad/mul_2Mul/gradients/mul_5_grad/tuple/control_dependency_1pow*
T0*
_output_shapes
: 
w
gradients/pow_grad/mul_3Mulgradients/pow_grad/mul_2gradients/pow_grad/Select_1*
T0*
_output_shapes
: 
£
gradients/pow_grad/Sum_1Sumgradients/pow_grad/mul_3*gradients/pow_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

gradients/pow_grad/Reshape_1Reshapegradients/pow_grad/Sum_1gradients/pow_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
g
#gradients/pow_grad/tuple/group_depsNoOp^gradients/pow_grad/Reshape^gradients/pow_grad/Reshape_1
É
+gradients/pow_grad/tuple/control_dependencyIdentitygradients/pow_grad/Reshape$^gradients/pow_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/pow_grad/Reshape*
_output_shapes
: 
Ļ
-gradients/pow_grad/tuple/control_dependency_1Identitygradients/pow_grad/Reshape_1$^gradients/pow_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/pow_grad/Reshape_1*
_output_shapes
: 
]
gradients/pow_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
_
gradients/pow_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
ŗ
*gradients/pow_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pow_1_grad/Shapegradients/pow_1_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
z
gradients/pow_1_grad/mulMul/gradients/mul_6_grad/tuple/control_dependency_1pow_1/y*
T0*
_output_shapes
: 
_
gradients/pow_1_grad/sub/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
e
gradients/pow_1_grad/subSubpow_1/ygradients/pow_1_grad/sub/y*
_output_shapes
: *
T0
a
gradients/pow_1_grad/PowPowSum_1gradients/pow_1_grad/sub*
T0*
_output_shapes
: 
v
gradients/pow_1_grad/mul_1Mulgradients/pow_1_grad/mulgradients/pow_1_grad/Pow*
_output_shapes
: *
T0
„
gradients/pow_1_grad/SumSumgradients/pow_1_grad/mul_1*gradients/pow_1_grad/BroadcastGradientArgs*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

gradients/pow_1_grad/ReshapeReshapegradients/pow_1_grad/Sumgradients/pow_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
c
gradients/pow_1_grad/Greater/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
o
gradients/pow_1_grad/GreaterGreaterSum_1gradients/pow_1_grad/Greater/y*
T0*
_output_shapes
: 
g
$gradients/pow_1_grad/ones_like/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
i
$gradients/pow_1_grad/ones_like/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
„
gradients/pow_1_grad/ones_likeFill$gradients/pow_1_grad/ones_like/Shape$gradients/pow_1_grad/ones_like/Const*
T0*

index_type0*
_output_shapes
: 

gradients/pow_1_grad/SelectSelectgradients/pow_1_grad/GreaterSum_1gradients/pow_1_grad/ones_like*
_output_shapes
: *
T0
]
gradients/pow_1_grad/LogLoggradients/pow_1_grad/Select*
_output_shapes
: *
T0
d
gradients/pow_1_grad/zeros_likeConst*
dtype0*
_output_shapes
: *
valueB
 *    
”
gradients/pow_1_grad/Select_1Selectgradients/pow_1_grad/Greatergradients/pow_1_grad/Loggradients/pow_1_grad/zeros_like*
T0*
_output_shapes
: 
z
gradients/pow_1_grad/mul_2Mul/gradients/mul_6_grad/tuple/control_dependency_1pow_1*
T0*
_output_shapes
: 
}
gradients/pow_1_grad/mul_3Mulgradients/pow_1_grad/mul_2gradients/pow_1_grad/Select_1*
T0*
_output_shapes
: 
©
gradients/pow_1_grad/Sum_1Sumgradients/pow_1_grad/mul_3,gradients/pow_1_grad/BroadcastGradientArgs:1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

gradients/pow_1_grad/Reshape_1Reshapegradients/pow_1_grad/Sum_1gradients/pow_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
m
%gradients/pow_1_grad/tuple/group_depsNoOp^gradients/pow_1_grad/Reshape^gradients/pow_1_grad/Reshape_1
Ń
-gradients/pow_1_grad/tuple/control_dependencyIdentitygradients/pow_1_grad/Reshape&^gradients/pow_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/pow_1_grad/Reshape*
_output_shapes
: 
×
/gradients/pow_1_grad/tuple/control_dependency_1Identitygradients/pow_1_grad/Reshape_1&^gradients/pow_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/pow_1_grad/Reshape_1*
_output_shapes
: 
j
 gradients/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
§
gradients/Sum_grad/ReshapeReshape+gradients/pow_grad/tuple/control_dependency gradients/Sum_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
]
gradients/Sum_grad/ShapeShapemul_2*
T0*
out_type0*
_output_shapes
:

gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/Shape*#
_output_shapes
:’’’’’’’’’*

Tmultiples0*
T0
l
"gradients/Sum_1_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
­
gradients/Sum_1_grad/ReshapeReshape-gradients/pow_1_grad/tuple/control_dependency"gradients/Sum_1_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
_
gradients/Sum_1_grad/ShapeShapemul_4*
T0*
out_type0*
_output_shapes
:

gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/Shape*
T0*#
_output_shapes
:’’’’’’’’’*

Tmultiples0
a
gradients/mul_2_grad/ShapeShapeSoftmax*
T0*
out_type0*
_output_shapes
:
a
gradients/mul_2_grad/Shape_1ShapeAbs_1*
_output_shapes
:*
T0*
out_type0
ŗ
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
m
gradients/mul_2_grad/MulMulgradients/Sum_grad/TileAbs_1*
T0*#
_output_shapes
:’’’’’’’’’
„
gradients/mul_2_grad/SumSumgradients/mul_2_grad/Mul*gradients/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:’’’’’’’’’
q
gradients/mul_2_grad/Mul_1MulSoftmaxgradients/Sum_grad/Tile*#
_output_shapes
:’’’’’’’’’*
T0
«
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/Mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:’’’’’’’’’
m
%gradients/mul_2_grad/tuple/group_depsNoOp^gradients/mul_2_grad/Reshape^gradients/mul_2_grad/Reshape_1
Ž
-gradients/mul_2_grad/tuple/control_dependencyIdentitygradients/mul_2_grad/Reshape&^gradients/mul_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_2_grad/Reshape*#
_output_shapes
:’’’’’’’’’
ä
/gradients/mul_2_grad/tuple/control_dependency_1Identitygradients/mul_2_grad/Reshape_1&^gradients/mul_2_grad/tuple/group_deps*#
_output_shapes
:’’’’’’’’’*
T0*1
_class'
%#loc:@gradients/mul_2_grad/Reshape_1
c
gradients/mul_4_grad/ShapeShape	Softmax_1*
out_type0*
_output_shapes
:*
T0
c
gradients/mul_4_grad/Shape_1ShapeMinimum*
_output_shapes
:*
T0*
out_type0
ŗ
*gradients/mul_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_4_grad/Shapegradients/mul_4_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
q
gradients/mul_4_grad/MulMulgradients/Sum_1_grad/TileMinimum*
T0*#
_output_shapes
:’’’’’’’’’
„
gradients/mul_4_grad/SumSumgradients/mul_4_grad/Mul*gradients/mul_4_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/mul_4_grad/ReshapeReshapegradients/mul_4_grad/Sumgradients/mul_4_grad/Shape*
T0*
Tshape0*#
_output_shapes
:’’’’’’’’’
u
gradients/mul_4_grad/Mul_1Mul	Softmax_1gradients/Sum_1_grad/Tile*#
_output_shapes
:’’’’’’’’’*
T0
«
gradients/mul_4_grad/Sum_1Sumgradients/mul_4_grad/Mul_1,gradients/mul_4_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/mul_4_grad/Reshape_1Reshapegradients/mul_4_grad/Sum_1gradients/mul_4_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:’’’’’’’’’
m
%gradients/mul_4_grad/tuple/group_depsNoOp^gradients/mul_4_grad/Reshape^gradients/mul_4_grad/Reshape_1
Ž
-gradients/mul_4_grad/tuple/control_dependencyIdentitygradients/mul_4_grad/Reshape&^gradients/mul_4_grad/tuple/group_deps*/
_class%
#!loc:@gradients/mul_4_grad/Reshape*#
_output_shapes
:’’’’’’’’’*
T0
ä
/gradients/mul_4_grad/tuple/control_dependency_1Identitygradients/mul_4_grad/Reshape_1&^gradients/mul_4_grad/tuple/group_deps*#
_output_shapes
:’’’’’’’’’*
T0*1
_class'
%#loc:@gradients/mul_4_grad/Reshape_1

gradients/Softmax_grad/mulMul-gradients/mul_2_grad/tuple/control_dependencySoftmax*
T0*#
_output_shapes
:’’’’’’’’’
w
,gradients/Softmax_grad/Sum/reduction_indicesConst*
valueB :
’’’’’’’’’*
dtype0*
_output_shapes
: 
­
gradients/Softmax_grad/SumSumgradients/Softmax_grad/mul,gradients/Softmax_grad/Sum/reduction_indices*
T0*
_output_shapes
:*
	keep_dims(*

Tidx0

gradients/Softmax_grad/subSub-gradients/mul_2_grad/tuple/control_dependencygradients/Softmax_grad/Sum*#
_output_shapes
:’’’’’’’’’*
T0
v
gradients/Softmax_grad/mul_1Mulgradients/Softmax_grad/subSoftmax*
T0*#
_output_shapes
:’’’’’’’’’

gradients/Softmax_1_grad/mulMul-gradients/mul_4_grad/tuple/control_dependency	Softmax_1*#
_output_shapes
:’’’’’’’’’*
T0
y
.gradients/Softmax_1_grad/Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB :
’’’’’’’’’
³
gradients/Softmax_1_grad/SumSumgradients/Softmax_1_grad/mul.gradients/Softmax_1_grad/Sum/reduction_indices*
T0*
_output_shapes
:*
	keep_dims(*

Tidx0

gradients/Softmax_1_grad/subSub-gradients/mul_4_grad/tuple/control_dependencygradients/Softmax_1_grad/Sum*#
_output_shapes
:’’’’’’’’’*
T0
|
gradients/Softmax_1_grad/mul_1Mulgradients/Softmax_1_grad/sub	Softmax_1*#
_output_shapes
:’’’’’’’’’*
T0
]
gradients/mul_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
a
gradients/mul_1_grad/Shape_1ShapeAbs_1*
T0*
out_type0*
_output_shapes
:
ŗ
*gradients/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_1_grad/Shapegradients/mul_1_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
r
gradients/mul_1_grad/MulMulgradients/Softmax_grad/mul_1Abs_1*
T0*#
_output_shapes
:’’’’’’’’’
„
gradients/mul_1_grad/SumSumgradients/mul_1_grad/Mul*gradients/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/mul_1_grad/ReshapeReshapegradients/mul_1_grad/Sumgradients/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
v
gradients/mul_1_grad/Mul_1Mulmul_1/xgradients/Softmax_grad/mul_1*#
_output_shapes
:’’’’’’’’’*
T0
«
gradients/mul_1_grad/Sum_1Sumgradients/mul_1_grad/Mul_1,gradients/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/mul_1_grad/Reshape_1Reshapegradients/mul_1_grad/Sum_1gradients/mul_1_grad/Shape_1*
Tshape0*#
_output_shapes
:’’’’’’’’’*
T0
m
%gradients/mul_1_grad/tuple/group_depsNoOp^gradients/mul_1_grad/Reshape^gradients/mul_1_grad/Reshape_1
Ń
-gradients/mul_1_grad/tuple/control_dependencyIdentitygradients/mul_1_grad/Reshape&^gradients/mul_1_grad/tuple/group_deps*/
_class%
#!loc:@gradients/mul_1_grad/Reshape*
_output_shapes
: *
T0
ä
/gradients/mul_1_grad/tuple/control_dependency_1Identitygradients/mul_1_grad/Reshape_1&^gradients/mul_1_grad/tuple/group_deps*#
_output_shapes
:’’’’’’’’’*
T0*1
_class'
%#loc:@gradients/mul_1_grad/Reshape_1
]
gradients/mul_3_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0
_
gradients/mul_3_grad/Shape_1ShapeAbs*
T0*
out_type0*
_output_shapes
:
ŗ
*gradients/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_3_grad/Shapegradients/mul_3_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
r
gradients/mul_3_grad/MulMulgradients/Softmax_1_grad/mul_1Abs*#
_output_shapes
:’’’’’’’’’*
T0
„
gradients/mul_3_grad/SumSumgradients/mul_3_grad/Mul*gradients/mul_3_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

gradients/mul_3_grad/ReshapeReshapegradients/mul_3_grad/Sumgradients/mul_3_grad/Shape*
_output_shapes
: *
T0*
Tshape0
x
gradients/mul_3_grad/Mul_1Mulmul_3/xgradients/Softmax_1_grad/mul_1*
T0*#
_output_shapes
:’’’’’’’’’
«
gradients/mul_3_grad/Sum_1Sumgradients/mul_3_grad/Mul_1,gradients/mul_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/mul_3_grad/Reshape_1Reshapegradients/mul_3_grad/Sum_1gradients/mul_3_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:’’’’’’’’’
m
%gradients/mul_3_grad/tuple/group_depsNoOp^gradients/mul_3_grad/Reshape^gradients/mul_3_grad/Reshape_1
Ń
-gradients/mul_3_grad/tuple/control_dependencyIdentitygradients/mul_3_grad/Reshape&^gradients/mul_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_3_grad/Reshape*
_output_shapes
: 
ä
/gradients/mul_3_grad/tuple/control_dependency_1Identitygradients/mul_3_grad/Reshape_1&^gradients/mul_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_3_grad/Reshape_1*#
_output_shapes
:’’’’’’’’’
ā
gradients/AddNAddN/gradients/mul_2_grad/tuple/control_dependency_1/gradients/mul_1_grad/tuple/control_dependency_1*
T0*1
_class'
%#loc:@gradients/mul_2_grad/Reshape_1*
N*#
_output_shapes
:’’’’’’’’’
T
gradients/Abs_1_grad/SignSignsub*#
_output_shapes
:’’’’’’’’’*
T0
x
gradients/Abs_1_grad/mulMulgradients/AddNgradients/Abs_1_grad/Sign*#
_output_shapes
:’’’’’’’’’*
T0
V
gradients/Abs_grad/SignSignMinimum*
T0*#
_output_shapes
:’’’’’’’’’

gradients/Abs_grad/mulMul/gradients/mul_3_grad/tuple/control_dependency_1gradients/Abs_grad/Sign*
T0*#
_output_shapes
:’’’’’’’’’
Ė
gradients/AddN_1AddN/gradients/mul_4_grad/tuple/control_dependency_1gradients/Abs_grad/mul*
T0*1
_class'
%#loc:@gradients/mul_4_grad/Reshape_1*
N*#
_output_shapes
:’’’’’’’’’
_
gradients/Minimum_grad/ShapeShapemul*
out_type0*
_output_shapes
:*
T0
a
gradients/Minimum_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
n
gradients/Minimum_grad/Shape_2Shapegradients/AddN_1*
T0*
out_type0*
_output_shapes
:
g
"gradients/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ø
gradients/Minimum_grad/zerosFillgradients/Minimum_grad/Shape_2"gradients/Minimum_grad/zeros/Const*
T0*

index_type0*#
_output_shapes
:’’’’’’’’’
k
 gradients/Minimum_grad/LessEqual	LessEqualmul	Minimum/y*#
_output_shapes
:’’’’’’’’’*
T0
Ą
,gradients/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Minimum_grad/Shapegradients/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
§
gradients/Minimum_grad/SelectSelect gradients/Minimum_grad/LessEqualgradients/AddN_1gradients/Minimum_grad/zeros*
T0*#
_output_shapes
:’’’’’’’’’
®
gradients/Minimum_grad/SumSumgradients/Minimum_grad/Select,gradients/Minimum_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/Minimum_grad/ReshapeReshapegradients/Minimum_grad/Sumgradients/Minimum_grad/Shape*
T0*
Tshape0*#
_output_shapes
:’’’’’’’’’
©
gradients/Minimum_grad/Select_1Select gradients/Minimum_grad/LessEqualgradients/Minimum_grad/zerosgradients/AddN_1*
T0*#
_output_shapes
:’’’’’’’’’
“
gradients/Minimum_grad/Sum_1Sumgradients/Minimum_grad/Select_1.gradients/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

 gradients/Minimum_grad/Reshape_1Reshapegradients/Minimum_grad/Sum_1gradients/Minimum_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
s
'gradients/Minimum_grad/tuple/group_depsNoOp^gradients/Minimum_grad/Reshape!^gradients/Minimum_grad/Reshape_1
ę
/gradients/Minimum_grad/tuple/control_dependencyIdentitygradients/Minimum_grad/Reshape(^gradients/Minimum_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Minimum_grad/Reshape*#
_output_shapes
:’’’’’’’’’*
T0
ß
1gradients/Minimum_grad/tuple/control_dependency_1Identity gradients/Minimum_grad/Reshape_1(^gradients/Minimum_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/Minimum_grad/Reshape_1*
_output_shapes
: 
Ķ
gradients/AddN_2AddNgradients/Mean_grad/truediv/gradients/Minimum_grad/tuple/control_dependency*
N*#
_output_shapes
:’’’’’’’’’*
T0*.
_class$
" loc:@gradients/Mean_grad/truediv
[
gradients/mul_grad/ShapeShapesub*
T0*
out_type0*
_output_shapes
:
g
gradients/mul_grad/Shape_1ShapePlaceholder_3*
out_type0*
_output_shapes
:*
T0
“
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
l
gradients/mul_grad/MulMulgradients/AddN_2Placeholder_3*#
_output_shapes
:’’’’’’’’’*
T0

gradients/mul_grad/SumSumgradients/mul_grad/Mul(gradients/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:’’’’’’’’’
d
gradients/mul_grad/Mul_1Mulsubgradients/AddN_2*#
_output_shapes
:’’’’’’’’’*
T0
„
gradients/mul_grad/Sum_1Sumgradients/mul_grad/Mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:’’’’’’’’’
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
Ö
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape*#
_output_shapes
:’’’’’’’’’
Ü
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*#
_output_shapes
:’’’’’’’’’
Ć
gradients/AddN_3AddNgradients/Abs_1_grad/mul+gradients/mul_grad/tuple/control_dependency*
T0*+
_class!
loc:@gradients/Abs_1_grad/mul*
N*#
_output_shapes
:’’’’’’’’’
`
gradients/sub_grad/ShapeShapepi/sub_5*
out_type0*
_output_shapes
:*
T0
g
gradients/sub_grad/Shape_1ShapePlaceholder_4*
T0*
out_type0*
_output_shapes
:
“
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0

gradients/sub_grad/SumSumgradients/AddN_3(gradients/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*#
_output_shapes
:’’’’’’’’’

gradients/sub_grad/Sum_1Sumgradients/AddN_3*gradients/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_output_shapes
:*
T0

gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:’’’’’’’’’
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
Ö
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*#
_output_shapes
:’’’’’’’’’
Ü
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*#
_output_shapes
:’’’’’’’’’
e
gradients/pi/sub_5_grad/ShapeShapepi/Sum_2*
T0*
out_type0*
_output_shapes
:
g
gradients/pi/sub_5_grad/Shape_1Shapepi/Sum_3*
T0*
out_type0*
_output_shapes
:
Ć
-gradients/pi/sub_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/sub_5_grad/Shapegradients/pi/sub_5_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
¾
gradients/pi/sub_5_grad/SumSum+gradients/sub_grad/tuple/control_dependency-gradients/pi/sub_5_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
¢
gradients/pi/sub_5_grad/ReshapeReshapegradients/pi/sub_5_grad/Sumgradients/pi/sub_5_grad/Shape*
T0*
Tshape0*#
_output_shapes
:’’’’’’’’’
Ā
gradients/pi/sub_5_grad/Sum_1Sum+gradients/sub_grad/tuple/control_dependency/gradients/pi/sub_5_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
d
gradients/pi/sub_5_grad/NegNeggradients/pi/sub_5_grad/Sum_1*
_output_shapes
:*
T0
¦
!gradients/pi/sub_5_grad/Reshape_1Reshapegradients/pi/sub_5_grad/Neggradients/pi/sub_5_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:’’’’’’’’’
v
(gradients/pi/sub_5_grad/tuple/group_depsNoOp ^gradients/pi/sub_5_grad/Reshape"^gradients/pi/sub_5_grad/Reshape_1
ź
0gradients/pi/sub_5_grad/tuple/control_dependencyIdentitygradients/pi/sub_5_grad/Reshape)^gradients/pi/sub_5_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/pi/sub_5_grad/Reshape*#
_output_shapes
:’’’’’’’’’
š
2gradients/pi/sub_5_grad/tuple/control_dependency_1Identity!gradients/pi/sub_5_grad/Reshape_1)^gradients/pi/sub_5_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/pi/sub_5_grad/Reshape_1*#
_output_shapes
:’’’’’’’’’
u
gradients/pi/Sum_2_grad/ShapeShapepi/Normal/log_prob_1/sub*
T0*
out_type0*
_output_shapes
:

gradients/pi/Sum_2_grad/SizeConst*
value	B :*0
_class&
$"loc:@gradients/pi/Sum_2_grad/Shape*
dtype0*
_output_shapes
: 
Æ
gradients/pi/Sum_2_grad/addAddpi/Sum_2/reduction_indicesgradients/pi/Sum_2_grad/Size*
T0*0
_class&
$"loc:@gradients/pi/Sum_2_grad/Shape*
_output_shapes
: 
µ
gradients/pi/Sum_2_grad/modFloorModgradients/pi/Sum_2_grad/addgradients/pi/Sum_2_grad/Size*
T0*0
_class&
$"loc:@gradients/pi/Sum_2_grad/Shape*
_output_shapes
: 

gradients/pi/Sum_2_grad/Shape_1Const*
valueB *0
_class&
$"loc:@gradients/pi/Sum_2_grad/Shape*
dtype0*
_output_shapes
: 

#gradients/pi/Sum_2_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *0
_class&
$"loc:@gradients/pi/Sum_2_grad/Shape

#gradients/pi/Sum_2_grad/range/deltaConst*
value	B :*0
_class&
$"loc:@gradients/pi/Sum_2_grad/Shape*
dtype0*
_output_shapes
: 
č
gradients/pi/Sum_2_grad/rangeRange#gradients/pi/Sum_2_grad/range/startgradients/pi/Sum_2_grad/Size#gradients/pi/Sum_2_grad/range/delta*
_output_shapes
:*

Tidx0*0
_class&
$"loc:@gradients/pi/Sum_2_grad/Shape

"gradients/pi/Sum_2_grad/Fill/valueConst*
value	B :*0
_class&
$"loc:@gradients/pi/Sum_2_grad/Shape*
dtype0*
_output_shapes
: 
Ī
gradients/pi/Sum_2_grad/FillFillgradients/pi/Sum_2_grad/Shape_1"gradients/pi/Sum_2_grad/Fill/value*
T0*

index_type0*0
_class&
$"loc:@gradients/pi/Sum_2_grad/Shape*
_output_shapes
: 

%gradients/pi/Sum_2_grad/DynamicStitchDynamicStitchgradients/pi/Sum_2_grad/rangegradients/pi/Sum_2_grad/modgradients/pi/Sum_2_grad/Shapegradients/pi/Sum_2_grad/Fill*
T0*0
_class&
$"loc:@gradients/pi/Sum_2_grad/Shape*
N*
_output_shapes
:

!gradients/pi/Sum_2_grad/Maximum/yConst*
value	B :*0
_class&
$"loc:@gradients/pi/Sum_2_grad/Shape*
dtype0*
_output_shapes
: 
Ė
gradients/pi/Sum_2_grad/MaximumMaximum%gradients/pi/Sum_2_grad/DynamicStitch!gradients/pi/Sum_2_grad/Maximum/y*
T0*0
_class&
$"loc:@gradients/pi/Sum_2_grad/Shape*
_output_shapes
:
Ć
 gradients/pi/Sum_2_grad/floordivFloorDivgradients/pi/Sum_2_grad/Shapegradients/pi/Sum_2_grad/Maximum*
_output_shapes
:*
T0*0
_class&
$"loc:@gradients/pi/Sum_2_grad/Shape
Ģ
gradients/pi/Sum_2_grad/ReshapeReshape0gradients/pi/sub_5_grad/tuple/control_dependency%gradients/pi/Sum_2_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
«
gradients/pi/Sum_2_grad/TileTilegradients/pi/Sum_2_grad/Reshape gradients/pi/Sum_2_grad/floordiv*'
_output_shapes
:’’’’’’’’’*

Tmultiples0*
T0

-gradients/pi/Normal/log_prob_1/sub_grad/ShapeShapepi/Normal/log_prob_1/mul*
_output_shapes
:*
T0*
out_type0

/gradients/pi/Normal/log_prob_1/sub_grad/Shape_1Shapepi/Normal/log_prob_1/add*
T0*
out_type0*
_output_shapes
:
ó
=gradients/pi/Normal/log_prob_1/sub_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/pi/Normal/log_prob_1/sub_grad/Shape/gradients/pi/Normal/log_prob_1/sub_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
Ļ
+gradients/pi/Normal/log_prob_1/sub_grad/SumSumgradients/pi/Sum_2_grad/Tile=gradients/pi/Normal/log_prob_1/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ö
/gradients/pi/Normal/log_prob_1/sub_grad/ReshapeReshape+gradients/pi/Normal/log_prob_1/sub_grad/Sum-gradients/pi/Normal/log_prob_1/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’
Ó
-gradients/pi/Normal/log_prob_1/sub_grad/Sum_1Sumgradients/pi/Sum_2_grad/Tile?gradients/pi/Normal/log_prob_1/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

+gradients/pi/Normal/log_prob_1/sub_grad/NegNeg-gradients/pi/Normal/log_prob_1/sub_grad/Sum_1*
T0*
_output_shapes
:
Ś
1gradients/pi/Normal/log_prob_1/sub_grad/Reshape_1Reshape+gradients/pi/Normal/log_prob_1/sub_grad/Neg/gradients/pi/Normal/log_prob_1/sub_grad/Shape_1*'
_output_shapes
:’’’’’’’’’*
T0*
Tshape0
¦
8gradients/pi/Normal/log_prob_1/sub_grad/tuple/group_depsNoOp0^gradients/pi/Normal/log_prob_1/sub_grad/Reshape2^gradients/pi/Normal/log_prob_1/sub_grad/Reshape_1
®
@gradients/pi/Normal/log_prob_1/sub_grad/tuple/control_dependencyIdentity/gradients/pi/Normal/log_prob_1/sub_grad/Reshape9^gradients/pi/Normal/log_prob_1/sub_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/pi/Normal/log_prob_1/sub_grad/Reshape*'
_output_shapes
:’’’’’’’’’
“
Bgradients/pi/Normal/log_prob_1/sub_grad/tuple/control_dependency_1Identity1gradients/pi/Normal/log_prob_1/sub_grad/Reshape_19^gradients/pi/Normal/log_prob_1/sub_grad/tuple/group_deps*'
_output_shapes
:’’’’’’’’’*
T0*D
_class:
86loc:@gradients/pi/Normal/log_prob_1/sub_grad/Reshape_1
p
-gradients/pi/Normal/log_prob_1/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 

/gradients/pi/Normal/log_prob_1/mul_grad/Shape_1Shapepi/Normal/log_prob_1/Square*
out_type0*
_output_shapes
:*
T0
ó
=gradients/pi/Normal/log_prob_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/pi/Normal/log_prob_1/mul_grad/Shape/gradients/pi/Normal/log_prob_1/mul_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
Ć
+gradients/pi/Normal/log_prob_1/mul_grad/MulMul@gradients/pi/Normal/log_prob_1/sub_grad/tuple/control_dependencypi/Normal/log_prob_1/Square*'
_output_shapes
:’’’’’’’’’*
T0
Ž
+gradients/pi/Normal/log_prob_1/mul_grad/SumSum+gradients/pi/Normal/log_prob_1/mul_grad/Mul=gradients/pi/Normal/log_prob_1/mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Å
/gradients/pi/Normal/log_prob_1/mul_grad/ReshapeReshape+gradients/pi/Normal/log_prob_1/mul_grad/Sum-gradients/pi/Normal/log_prob_1/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Ä
-gradients/pi/Normal/log_prob_1/mul_grad/Mul_1Mulpi/Normal/log_prob_1/mul/x@gradients/pi/Normal/log_prob_1/sub_grad/tuple/control_dependency*'
_output_shapes
:’’’’’’’’’*
T0
ä
-gradients/pi/Normal/log_prob_1/mul_grad/Sum_1Sum-gradients/pi/Normal/log_prob_1/mul_grad/Mul_1?gradients/pi/Normal/log_prob_1/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ü
1gradients/pi/Normal/log_prob_1/mul_grad/Reshape_1Reshape-gradients/pi/Normal/log_prob_1/mul_grad/Sum_1/gradients/pi/Normal/log_prob_1/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’
¦
8gradients/pi/Normal/log_prob_1/mul_grad/tuple/group_depsNoOp0^gradients/pi/Normal/log_prob_1/mul_grad/Reshape2^gradients/pi/Normal/log_prob_1/mul_grad/Reshape_1

@gradients/pi/Normal/log_prob_1/mul_grad/tuple/control_dependencyIdentity/gradients/pi/Normal/log_prob_1/mul_grad/Reshape9^gradients/pi/Normal/log_prob_1/mul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/pi/Normal/log_prob_1/mul_grad/Reshape*
_output_shapes
: 
“
Bgradients/pi/Normal/log_prob_1/mul_grad/tuple/control_dependency_1Identity1gradients/pi/Normal/log_prob_1/mul_grad/Reshape_19^gradients/pi/Normal/log_prob_1/mul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/pi/Normal/log_prob_1/mul_grad/Reshape_1*'
_output_shapes
:’’’’’’’’’
p
-gradients/pi/Normal/log_prob_1/add_grad/ShapeConst*
_output_shapes
: *
valueB *
dtype0

/gradients/pi/Normal/log_prob_1/add_grad/Shape_1Shapepi/Normal/log_prob_1/Log*
_output_shapes
:*
T0*
out_type0
ó
=gradients/pi/Normal/log_prob_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/pi/Normal/log_prob_1/add_grad/Shape/gradients/pi/Normal/log_prob_1/add_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
õ
+gradients/pi/Normal/log_prob_1/add_grad/SumSumBgradients/pi/Normal/log_prob_1/sub_grad/tuple/control_dependency_1=gradients/pi/Normal/log_prob_1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Å
/gradients/pi/Normal/log_prob_1/add_grad/ReshapeReshape+gradients/pi/Normal/log_prob_1/add_grad/Sum-gradients/pi/Normal/log_prob_1/add_grad/Shape*
_output_shapes
: *
T0*
Tshape0
ł
-gradients/pi/Normal/log_prob_1/add_grad/Sum_1SumBgradients/pi/Normal/log_prob_1/sub_grad/tuple/control_dependency_1?gradients/pi/Normal/log_prob_1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ü
1gradients/pi/Normal/log_prob_1/add_grad/Reshape_1Reshape-gradients/pi/Normal/log_prob_1/add_grad/Sum_1/gradients/pi/Normal/log_prob_1/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’
¦
8gradients/pi/Normal/log_prob_1/add_grad/tuple/group_depsNoOp0^gradients/pi/Normal/log_prob_1/add_grad/Reshape2^gradients/pi/Normal/log_prob_1/add_grad/Reshape_1

@gradients/pi/Normal/log_prob_1/add_grad/tuple/control_dependencyIdentity/gradients/pi/Normal/log_prob_1/add_grad/Reshape9^gradients/pi/Normal/log_prob_1/add_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/pi/Normal/log_prob_1/add_grad/Reshape*
_output_shapes
: 
“
Bgradients/pi/Normal/log_prob_1/add_grad/tuple/control_dependency_1Identity1gradients/pi/Normal/log_prob_1/add_grad/Reshape_19^gradients/pi/Normal/log_prob_1/add_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/pi/Normal/log_prob_1/add_grad/Reshape_1*'
_output_shapes
:’’’’’’’’’
ŗ
0gradients/pi/Normal/log_prob_1/Square_grad/ConstConstC^gradients/pi/Normal/log_prob_1/mul_grad/tuple/control_dependency_1*
valueB
 *   @*
dtype0*
_output_shapes
: 
Ć
.gradients/pi/Normal/log_prob_1/Square_grad/MulMul(pi/Normal/log_prob_1/standardize/truediv0gradients/pi/Normal/log_prob_1/Square_grad/Const*
T0*'
_output_shapes
:’’’’’’’’’
Ż
0gradients/pi/Normal/log_prob_1/Square_grad/Mul_1MulBgradients/pi/Normal/log_prob_1/mul_grad/tuple/control_dependency_1.gradients/pi/Normal/log_prob_1/Square_grad/Mul*
T0*'
_output_shapes
:’’’’’’’’’
Ķ
2gradients/pi/Normal/log_prob_1/Log_grad/Reciprocal
Reciprocalpi/Normal/Identity_1C^gradients/pi/Normal/log_prob_1/add_grad/tuple/control_dependency_1*'
_output_shapes
:’’’’’’’’’*
T0
Ü
+gradients/pi/Normal/log_prob_1/Log_grad/mulMulBgradients/pi/Normal/log_prob_1/add_grad/tuple/control_dependency_12gradients/pi/Normal/log_prob_1/Log_grad/Reciprocal*
T0*'
_output_shapes
:’’’’’’’’’
”
=gradients/pi/Normal/log_prob_1/standardize/truediv_grad/ShapeShape$pi/Normal/log_prob_1/standardize/sub*
T0*
out_type0*
_output_shapes
:

?gradients/pi/Normal/log_prob_1/standardize/truediv_grad/Shape_1Shapepi/Normal/Identity_1*
T0*
out_type0*
_output_shapes
:
£
Mgradients/pi/Normal/log_prob_1/standardize/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs=gradients/pi/Normal/log_prob_1/standardize/truediv_grad/Shape?gradients/pi/Normal/log_prob_1/standardize/truediv_grad/Shape_1*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’*
T0
Ä
?gradients/pi/Normal/log_prob_1/standardize/truediv_grad/RealDivRealDiv0gradients/pi/Normal/log_prob_1/Square_grad/Mul_1pi/Normal/Identity_1*
T0*'
_output_shapes
:’’’’’’’’’

;gradients/pi/Normal/log_prob_1/standardize/truediv_grad/SumSum?gradients/pi/Normal/log_prob_1/standardize/truediv_grad/RealDivMgradients/pi/Normal/log_prob_1/standardize/truediv_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

?gradients/pi/Normal/log_prob_1/standardize/truediv_grad/ReshapeReshape;gradients/pi/Normal/log_prob_1/standardize/truediv_grad/Sum=gradients/pi/Normal/log_prob_1/standardize/truediv_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’

;gradients/pi/Normal/log_prob_1/standardize/truediv_grad/NegNeg$pi/Normal/log_prob_1/standardize/sub*
T0*'
_output_shapes
:’’’’’’’’’
Ń
Agradients/pi/Normal/log_prob_1/standardize/truediv_grad/RealDiv_1RealDiv;gradients/pi/Normal/log_prob_1/standardize/truediv_grad/Negpi/Normal/Identity_1*'
_output_shapes
:’’’’’’’’’*
T0
×
Agradients/pi/Normal/log_prob_1/standardize/truediv_grad/RealDiv_2RealDivAgradients/pi/Normal/log_prob_1/standardize/truediv_grad/RealDiv_1pi/Normal/Identity_1*'
_output_shapes
:’’’’’’’’’*
T0
é
;gradients/pi/Normal/log_prob_1/standardize/truediv_grad/mulMul0gradients/pi/Normal/log_prob_1/Square_grad/Mul_1Agradients/pi/Normal/log_prob_1/standardize/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:’’’’’’’’’

=gradients/pi/Normal/log_prob_1/standardize/truediv_grad/Sum_1Sum;gradients/pi/Normal/log_prob_1/standardize/truediv_grad/mulOgradients/pi/Normal/log_prob_1/standardize/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

Agradients/pi/Normal/log_prob_1/standardize/truediv_grad/Reshape_1Reshape=gradients/pi/Normal/log_prob_1/standardize/truediv_grad/Sum_1?gradients/pi/Normal/log_prob_1/standardize/truediv_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’
Ö
Hgradients/pi/Normal/log_prob_1/standardize/truediv_grad/tuple/group_depsNoOp@^gradients/pi/Normal/log_prob_1/standardize/truediv_grad/ReshapeB^gradients/pi/Normal/log_prob_1/standardize/truediv_grad/Reshape_1
ī
Pgradients/pi/Normal/log_prob_1/standardize/truediv_grad/tuple/control_dependencyIdentity?gradients/pi/Normal/log_prob_1/standardize/truediv_grad/ReshapeI^gradients/pi/Normal/log_prob_1/standardize/truediv_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/pi/Normal/log_prob_1/standardize/truediv_grad/Reshape*'
_output_shapes
:’’’’’’’’’
ō
Rgradients/pi/Normal/log_prob_1/standardize/truediv_grad/tuple/control_dependency_1IdentityAgradients/pi/Normal/log_prob_1/standardize/truediv_grad/Reshape_1I^gradients/pi/Normal/log_prob_1/standardize/truediv_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/pi/Normal/log_prob_1/standardize/truediv_grad/Reshape_1*'
_output_shapes
:’’’’’’’’’

9gradients/pi/Normal/log_prob_1/standardize/sub_grad/ShapeShapePlaceholder_1*
T0*
out_type0*
_output_shapes
:

;gradients/pi/Normal/log_prob_1/standardize/sub_grad/Shape_1Shapepi/Normal/Identity*
T0*
out_type0*
_output_shapes
:

Igradients/pi/Normal/log_prob_1/standardize/sub_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients/pi/Normal/log_prob_1/standardize/sub_grad/Shape;gradients/pi/Normal/log_prob_1/standardize/sub_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

7gradients/pi/Normal/log_prob_1/standardize/sub_grad/SumSumPgradients/pi/Normal/log_prob_1/standardize/truediv_grad/tuple/control_dependencyIgradients/pi/Normal/log_prob_1/standardize/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
ś
;gradients/pi/Normal/log_prob_1/standardize/sub_grad/ReshapeReshape7gradients/pi/Normal/log_prob_1/standardize/sub_grad/Sum9gradients/pi/Normal/log_prob_1/standardize/sub_grad/Shape*'
_output_shapes
:’’’’’’’’’*
T0*
Tshape0

9gradients/pi/Normal/log_prob_1/standardize/sub_grad/Sum_1SumPgradients/pi/Normal/log_prob_1/standardize/truediv_grad/tuple/control_dependencyKgradients/pi/Normal/log_prob_1/standardize/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

7gradients/pi/Normal/log_prob_1/standardize/sub_grad/NegNeg9gradients/pi/Normal/log_prob_1/standardize/sub_grad/Sum_1*
_output_shapes
:*
T0
ž
=gradients/pi/Normal/log_prob_1/standardize/sub_grad/Reshape_1Reshape7gradients/pi/Normal/log_prob_1/standardize/sub_grad/Neg;gradients/pi/Normal/log_prob_1/standardize/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’
Ź
Dgradients/pi/Normal/log_prob_1/standardize/sub_grad/tuple/group_depsNoOp<^gradients/pi/Normal/log_prob_1/standardize/sub_grad/Reshape>^gradients/pi/Normal/log_prob_1/standardize/sub_grad/Reshape_1
Ž
Lgradients/pi/Normal/log_prob_1/standardize/sub_grad/tuple/control_dependencyIdentity;gradients/pi/Normal/log_prob_1/standardize/sub_grad/ReshapeE^gradients/pi/Normal/log_prob_1/standardize/sub_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/pi/Normal/log_prob_1/standardize/sub_grad/Reshape*'
_output_shapes
:’’’’’’’’’
ä
Ngradients/pi/Normal/log_prob_1/standardize/sub_grad/tuple/control_dependency_1Identity=gradients/pi/Normal/log_prob_1/standardize/sub_grad/Reshape_1E^gradients/pi/Normal/log_prob_1/standardize/sub_grad/tuple/group_deps*'
_output_shapes
:’’’’’’’’’*
T0*P
_classF
DBloc:@gradients/pi/Normal/log_prob_1/standardize/sub_grad/Reshape_1

gradients/AddN_4AddN+gradients/pi/Normal/log_prob_1/Log_grad/mulRgradients/pi/Normal/log_prob_1/standardize/truediv_grad/tuple/control_dependency_1*
T0*>
_class4
20loc:@gradients/pi/Normal/log_prob_1/Log_grad/mul*
N*'
_output_shapes
:’’’’’’’’’

6gradients/pi/actor/actor/dense_3/Softplus_grad/SigmoidSigmoidpi/actor/actor/dense_3/BiasAdd*
T0*'
_output_shapes
:’’’’’’’’’
µ
2gradients/pi/actor/actor/dense_3/Softplus_grad/mulMulgradients/AddN_46gradients/pi/actor/actor/dense_3/Softplus_grad/Sigmoid*'
_output_shapes
:’’’’’’’’’*
T0
Ō
9gradients/pi/actor/actor/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradNgradients/pi/Normal/log_prob_1/standardize/sub_grad/tuple/control_dependency_1*
data_formatNHWC*
_output_shapes
:*
T0
Ó
>gradients/pi/actor/actor/dense_2/BiasAdd_grad/tuple/group_depsNoOpO^gradients/pi/Normal/log_prob_1/standardize/sub_grad/tuple/control_dependency_1:^gradients/pi/actor/actor/dense_2/BiasAdd_grad/BiasAddGrad
ē
Fgradients/pi/actor/actor/dense_2/BiasAdd_grad/tuple/control_dependencyIdentityNgradients/pi/Normal/log_prob_1/standardize/sub_grad/tuple/control_dependency_1?^gradients/pi/actor/actor/dense_2/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/pi/Normal/log_prob_1/standardize/sub_grad/Reshape_1*'
_output_shapes
:’’’’’’’’’
Ć
Hgradients/pi/actor/actor/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity9gradients/pi/actor/actor/dense_2/BiasAdd_grad/BiasAddGrad?^gradients/pi/actor/actor/dense_2/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/pi/actor/actor/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
ø
9gradients/pi/actor/actor/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad2gradients/pi/actor/actor/dense_3/Softplus_grad/mul*
T0*
data_formatNHWC*
_output_shapes
:
·
>gradients/pi/actor/actor/dense_3/BiasAdd_grad/tuple/group_depsNoOp:^gradients/pi/actor/actor/dense_3/BiasAdd_grad/BiasAddGrad3^gradients/pi/actor/actor/dense_3/Softplus_grad/mul
Ą
Fgradients/pi/actor/actor/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity2gradients/pi/actor/actor/dense_3/Softplus_grad/mul?^gradients/pi/actor/actor/dense_3/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:’’’’’’’’’*
T0*E
_class;
97loc:@gradients/pi/actor/actor/dense_3/Softplus_grad/mul
Ć
Hgradients/pi/actor/actor/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity9gradients/pi/actor/actor/dense_3/BiasAdd_grad/BiasAddGrad?^gradients/pi/actor/actor/dense_3/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/pi/actor/actor/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

3gradients/pi/actor/actor/dense_2/MatMul_grad/MatMulMatMulFgradients/pi/actor/actor/dense_2/BiasAdd_grad/tuple/control_dependency,pi/actor/actor/dense_2/MatMul/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b(
’
5gradients/pi/actor/actor/dense_2/MatMul_grad/MatMul_1MatMul&pi/actor/actor/leaky_re_lu_1/LeakyReluFgradients/pi/actor/actor/dense_2/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0
³
=gradients/pi/actor/actor/dense_2/MatMul_grad/tuple/group_depsNoOp4^gradients/pi/actor/actor/dense_2/MatMul_grad/MatMul6^gradients/pi/actor/actor/dense_2/MatMul_grad/MatMul_1
Į
Egradients/pi/actor/actor/dense_2/MatMul_grad/tuple/control_dependencyIdentity3gradients/pi/actor/actor/dense_2/MatMul_grad/MatMul>^gradients/pi/actor/actor/dense_2/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/pi/actor/actor/dense_2/MatMul_grad/MatMul*(
_output_shapes
:’’’’’’’’’
¾
Ggradients/pi/actor/actor/dense_2/MatMul_grad/tuple/control_dependency_1Identity5gradients/pi/actor/actor/dense_2/MatMul_grad/MatMul_1>^gradients/pi/actor/actor/dense_2/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/pi/actor/actor/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	

3gradients/pi/actor/actor/dense_3/MatMul_grad/MatMulMatMulFgradients/pi/actor/actor/dense_3/BiasAdd_grad/tuple/control_dependency,pi/actor/actor/dense_3/MatMul/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b(
’
5gradients/pi/actor/actor/dense_3/MatMul_grad/MatMul_1MatMul&pi/actor/actor/leaky_re_lu_1/LeakyReluFgradients/pi/actor/actor/dense_3/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	*
transpose_a(*
transpose_b( *
T0
³
=gradients/pi/actor/actor/dense_3/MatMul_grad/tuple/group_depsNoOp4^gradients/pi/actor/actor/dense_3/MatMul_grad/MatMul6^gradients/pi/actor/actor/dense_3/MatMul_grad/MatMul_1
Į
Egradients/pi/actor/actor/dense_3/MatMul_grad/tuple/control_dependencyIdentity3gradients/pi/actor/actor/dense_3/MatMul_grad/MatMul>^gradients/pi/actor/actor/dense_3/MatMul_grad/tuple/group_deps*(
_output_shapes
:’’’’’’’’’*
T0*F
_class<
:8loc:@gradients/pi/actor/actor/dense_3/MatMul_grad/MatMul
¾
Ggradients/pi/actor/actor/dense_3/MatMul_grad/tuple/control_dependency_1Identity5gradients/pi/actor/actor/dense_3/MatMul_grad/MatMul_1>^gradients/pi/actor/actor/dense_3/MatMul_grad/tuple/group_deps*
_output_shapes
:	*
T0*H
_class>
<:loc:@gradients/pi/actor/actor/dense_3/MatMul_grad/MatMul_1
Ŗ
gradients/AddN_5AddNEgradients/pi/actor/actor/dense_2/MatMul_grad/tuple/control_dependencyEgradients/pi/actor/actor/dense_3/MatMul_grad/tuple/control_dependency*
N*(
_output_shapes
:’’’’’’’’’*
T0*F
_class<
:8loc:@gradients/pi/actor/actor/dense_2/MatMul_grad/MatMul
É
Cgradients/pi/actor/actor/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGradLeakyReluGradgradients/AddN_5pi/actor/actor/dense_1/BiasAdd*
T0*
alpha%>*(
_output_shapes
:’’’’’’’’’
Ź
9gradients/pi/actor/actor/dense_1/BiasAdd_grad/BiasAddGradBiasAddGradCgradients/pi/actor/actor/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*
data_formatNHWC*
_output_shapes	
:*
T0
Č
>gradients/pi/actor/actor/dense_1/BiasAdd_grad/tuple/group_depsNoOp:^gradients/pi/actor/actor/dense_1/BiasAdd_grad/BiasAddGradD^gradients/pi/actor/actor/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad
ć
Fgradients/pi/actor/actor/dense_1/BiasAdd_grad/tuple/control_dependencyIdentityCgradients/pi/actor/actor/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad?^gradients/pi/actor/actor/dense_1/BiasAdd_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/pi/actor/actor/leaky_re_lu_1/LeakyRelu_grad/LeakyReluGrad*(
_output_shapes
:’’’’’’’’’
Ä
Hgradients/pi/actor/actor/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity9gradients/pi/actor/actor/dense_1/BiasAdd_grad/BiasAddGrad?^gradients/pi/actor/actor/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*L
_classB
@>loc:@gradients/pi/actor/actor/dense_1/BiasAdd_grad/BiasAddGrad

3gradients/pi/actor/actor/dense_1/MatMul_grad/MatMulMatMulFgradients/pi/actor/actor/dense_1/BiasAdd_grad/tuple/control_dependency,pi/actor/actor/dense_1/MatMul/ReadVariableOp*
T0*(
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b(
ž
5gradients/pi/actor/actor/dense_1/MatMul_grad/MatMul_1MatMul$pi/actor/actor/leaky_re_lu/LeakyReluFgradients/pi/actor/actor/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(
³
=gradients/pi/actor/actor/dense_1/MatMul_grad/tuple/group_depsNoOp4^gradients/pi/actor/actor/dense_1/MatMul_grad/MatMul6^gradients/pi/actor/actor/dense_1/MatMul_grad/MatMul_1
Į
Egradients/pi/actor/actor/dense_1/MatMul_grad/tuple/control_dependencyIdentity3gradients/pi/actor/actor/dense_1/MatMul_grad/MatMul>^gradients/pi/actor/actor/dense_1/MatMul_grad/tuple/group_deps*F
_class<
:8loc:@gradients/pi/actor/actor/dense_1/MatMul_grad/MatMul*(
_output_shapes
:’’’’’’’’’*
T0
æ
Ggradients/pi/actor/actor/dense_1/MatMul_grad/tuple/control_dependency_1Identity5gradients/pi/actor/actor/dense_1/MatMul_grad/MatMul_1>^gradients/pi/actor/actor/dense_1/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/pi/actor/actor/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:

ś
Agradients/pi/actor/actor/leaky_re_lu/LeakyRelu_grad/LeakyReluGradLeakyReluGradEgradients/pi/actor/actor/dense_1/MatMul_grad/tuple/control_dependencypi/actor/actor/dense/BiasAdd*
T0*
alpha%>*(
_output_shapes
:’’’’’’’’’
Ę
7gradients/pi/actor/actor/dense/BiasAdd_grad/BiasAddGradBiasAddGradAgradients/pi/actor/actor/leaky_re_lu/LeakyRelu_grad/LeakyReluGrad*
_output_shapes	
:*
T0*
data_formatNHWC
Ā
<gradients/pi/actor/actor/dense/BiasAdd_grad/tuple/group_depsNoOp8^gradients/pi/actor/actor/dense/BiasAdd_grad/BiasAddGradB^gradients/pi/actor/actor/leaky_re_lu/LeakyRelu_grad/LeakyReluGrad
Ū
Dgradients/pi/actor/actor/dense/BiasAdd_grad/tuple/control_dependencyIdentityAgradients/pi/actor/actor/leaky_re_lu/LeakyRelu_grad/LeakyReluGrad=^gradients/pi/actor/actor/dense/BiasAdd_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/pi/actor/actor/leaky_re_lu/LeakyRelu_grad/LeakyReluGrad*(
_output_shapes
:’’’’’’’’’
¼
Fgradients/pi/actor/actor/dense/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/pi/actor/actor/dense/BiasAdd_grad/BiasAddGrad=^gradients/pi/actor/actor/dense/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/pi/actor/actor/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:

1gradients/pi/actor/actor/dense/MatMul_grad/MatMulMatMulDgradients/pi/actor/actor/dense/BiasAdd_grad/tuple/control_dependency*pi/actor/actor/dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b(
ą
3gradients/pi/actor/actor/dense/MatMul_grad/MatMul_1MatMulPlaceholderDgradients/pi/actor/actor/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 
­
;gradients/pi/actor/actor/dense/MatMul_grad/tuple/group_depsNoOp2^gradients/pi/actor/actor/dense/MatMul_grad/MatMul4^gradients/pi/actor/actor/dense/MatMul_grad/MatMul_1
ø
Cgradients/pi/actor/actor/dense/MatMul_grad/tuple/control_dependencyIdentity1gradients/pi/actor/actor/dense/MatMul_grad/MatMul<^gradients/pi/actor/actor/dense/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/pi/actor/actor/dense/MatMul_grad/MatMul*'
_output_shapes
:’’’’’’’’’
¶
Egradients/pi/actor/actor/dense/MatMul_grad/tuple/control_dependency_1Identity3gradients/pi/actor/actor/dense/MatMul_grad/MatMul_1<^gradients/pi/actor/actor/dense/MatMul_grad/tuple/group_deps*
_output_shapes
:	*
T0*F
_class<
:8loc:@gradients/pi/actor/actor/dense/MatMul_grad/MatMul_1
Ģ
global_norm/L2LossL2LossEgradients/pi/actor/actor/dense/MatMul_grad/tuple/control_dependency_1*
T0*F
_class<
:8loc:@gradients/pi/actor/actor/dense/MatMul_grad/MatMul_1*
_output_shapes
: 
Ó
global_norm/L2Loss_1L2LossFgradients/pi/actor/actor/dense/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
: *
T0*J
_class@
><loc:@gradients/pi/actor/actor/dense/BiasAdd_grad/BiasAddGrad
Ņ
global_norm/L2Loss_2L2LossGgradients/pi/actor/actor/dense_1/MatMul_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@gradients/pi/actor/actor/dense_1/MatMul_grad/MatMul_1*
_output_shapes
: 
×
global_norm/L2Loss_3L2LossHgradients/pi/actor/actor/dense_1/BiasAdd_grad/tuple/control_dependency_1*
T0*L
_classB
@>loc:@gradients/pi/actor/actor/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
Ņ
global_norm/L2Loss_4L2LossGgradients/pi/actor/actor/dense_2/MatMul_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@gradients/pi/actor/actor/dense_2/MatMul_grad/MatMul_1*
_output_shapes
: 
×
global_norm/L2Loss_5L2LossHgradients/pi/actor/actor/dense_2/BiasAdd_grad/tuple/control_dependency_1*
T0*L
_classB
@>loc:@gradients/pi/actor/actor/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
Ņ
global_norm/L2Loss_6L2LossGgradients/pi/actor/actor/dense_3/MatMul_grad/tuple/control_dependency_1*
T0*H
_class>
<:loc:@gradients/pi/actor/actor/dense_3/MatMul_grad/MatMul_1*
_output_shapes
: 
×
global_norm/L2Loss_7L2LossHgradients/pi/actor/actor/dense_3/BiasAdd_grad/tuple/control_dependency_1*
T0*L
_classB
@>loc:@gradients/pi/actor/actor/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 

global_norm/stackPackglobal_norm/L2Lossglobal_norm/L2Loss_1global_norm/L2Loss_2global_norm/L2Loss_3global_norm/L2Loss_4global_norm/L2Loss_5global_norm/L2Loss_6global_norm/L2Loss_7*
T0*

axis *
N*
_output_shapes
:
[
global_norm/ConstConst*
valueB: *
dtype0*
_output_shapes
:
z
global_norm/SumSumglobal_norm/stackglobal_norm/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
X
global_norm/Const_1Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
]
global_norm/mulMulglobal_norm/Sumglobal_norm/Const_1*
T0*
_output_shapes
: 
Q
global_norm/global_normSqrtglobal_norm/mul*
T0*
_output_shapes
: 
b
clip_by_global_norm/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

clip_by_global_norm/truedivRealDivclip_by_global_norm/truediv/xglobal_norm/global_norm*
T0*
_output_shapes
: 
^
clip_by_global_norm/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
clip_by_global_norm/truediv_1/yConst*
_output_shapes
: *
valueB
 *   ?*
dtype0

clip_by_global_norm/truediv_1RealDivclip_by_global_norm/Constclip_by_global_norm/truediv_1/y*
T0*
_output_shapes
: 

clip_by_global_norm/MinimumMinimumclip_by_global_norm/truedivclip_by_global_norm/truediv_1*
T0*
_output_shapes
: 
^
clip_by_global_norm/mul/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
w
clip_by_global_norm/mulMulclip_by_global_norm/mul/xclip_by_global_norm/Minimum*
_output_shapes
: *
T0
b
clip_by_global_norm/IsFiniteIsFiniteglobal_norm/global_norm*
_output_shapes
: *
T0
`
clip_by_global_norm/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *  Ą

clip_by_global_norm/SelectSelectclip_by_global_norm/IsFiniteclip_by_global_norm/mulclip_by_global_norm/Const_1*
_output_shapes
: *
T0
õ
clip_by_global_norm/mul_1MulEgradients/pi/actor/actor/dense/MatMul_grad/tuple/control_dependency_1clip_by_global_norm/Select*
T0*F
_class<
:8loc:@gradients/pi/actor/actor/dense/MatMul_grad/MatMul_1*
_output_shapes
:	
Ć
*clip_by_global_norm/clip_by_global_norm/_0Identityclip_by_global_norm/mul_1*
T0*F
_class<
:8loc:@gradients/pi/actor/actor/dense/MatMul_grad/MatMul_1*
_output_shapes
:	
ö
clip_by_global_norm/mul_2MulFgradients/pi/actor/actor/dense/BiasAdd_grad/tuple/control_dependency_1clip_by_global_norm/Select*
T0*J
_class@
><loc:@gradients/pi/actor/actor/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ć
*clip_by_global_norm/clip_by_global_norm/_1Identityclip_by_global_norm/mul_2*
T0*J
_class@
><loc:@gradients/pi/actor/actor/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ś
clip_by_global_norm/mul_3MulGgradients/pi/actor/actor/dense_1/MatMul_grad/tuple/control_dependency_1clip_by_global_norm/Select*
T0*H
_class>
<:loc:@gradients/pi/actor/actor/dense_1/MatMul_grad/MatMul_1* 
_output_shapes
:

Ę
*clip_by_global_norm/clip_by_global_norm/_2Identityclip_by_global_norm/mul_3* 
_output_shapes
:
*
T0*H
_class>
<:loc:@gradients/pi/actor/actor/dense_1/MatMul_grad/MatMul_1
ś
clip_by_global_norm/mul_4MulHgradients/pi/actor/actor/dense_1/BiasAdd_grad/tuple/control_dependency_1clip_by_global_norm/Select*
T0*L
_classB
@>loc:@gradients/pi/actor/actor/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Å
*clip_by_global_norm/clip_by_global_norm/_3Identityclip_by_global_norm/mul_4*
T0*L
_classB
@>loc:@gradients/pi/actor/actor/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ł
clip_by_global_norm/mul_5MulGgradients/pi/actor/actor/dense_2/MatMul_grad/tuple/control_dependency_1clip_by_global_norm/Select*
_output_shapes
:	*
T0*H
_class>
<:loc:@gradients/pi/actor/actor/dense_2/MatMul_grad/MatMul_1
Å
*clip_by_global_norm/clip_by_global_norm/_4Identityclip_by_global_norm/mul_5*
_output_shapes
:	*
T0*H
_class>
<:loc:@gradients/pi/actor/actor/dense_2/MatMul_grad/MatMul_1
ł
clip_by_global_norm/mul_6MulHgradients/pi/actor/actor/dense_2/BiasAdd_grad/tuple/control_dependency_1clip_by_global_norm/Select*
T0*L
_classB
@>loc:@gradients/pi/actor/actor/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Ä
*clip_by_global_norm/clip_by_global_norm/_5Identityclip_by_global_norm/mul_6*
_output_shapes
:*
T0*L
_classB
@>loc:@gradients/pi/actor/actor/dense_2/BiasAdd_grad/BiasAddGrad
ł
clip_by_global_norm/mul_7MulGgradients/pi/actor/actor/dense_3/MatMul_grad/tuple/control_dependency_1clip_by_global_norm/Select*
_output_shapes
:	*
T0*H
_class>
<:loc:@gradients/pi/actor/actor/dense_3/MatMul_grad/MatMul_1
Å
*clip_by_global_norm/clip_by_global_norm/_6Identityclip_by_global_norm/mul_7*
T0*H
_class>
<:loc:@gradients/pi/actor/actor/dense_3/MatMul_grad/MatMul_1*
_output_shapes
:	
ł
clip_by_global_norm/mul_8MulHgradients/pi/actor/actor/dense_3/BiasAdd_grad/tuple/control_dependency_1clip_by_global_norm/Select*
T0*L
_classB
@>loc:@gradients/pi/actor/actor/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Ä
*clip_by_global_norm/clip_by_global_norm/_7Identityclip_by_global_norm/mul_8*
_output_shapes
:*
T0*L
_classB
@>loc:@gradients/pi/actor/actor/dense_3/BiasAdd_grad/BiasAddGrad

%beta1_power/Initializer/initial_valueConst*
_output_shapes
: *,
_class"
 loc:@pi/actor/actor/dense/bias*
valueB
 *fff?*
dtype0
©
beta1_powerVarHandleOp*,
_class"
 loc:@pi/actor/actor/dense/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_namebeta1_power

,beta1_power/IsInitialized/VarIsInitializedOpVarIsInitializedOpbeta1_power*,
_class"
 loc:@pi/actor/actor/dense/bias*
_output_shapes
: 

beta1_power/AssignAssignVariableOpbeta1_power%beta1_power/Initializer/initial_value*
dtype0*,
_class"
 loc:@pi/actor/actor/dense/bias

beta1_power/Read/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *,
_class"
 loc:@pi/actor/actor/dense/bias*
dtype0

%beta2_power/Initializer/initial_valueConst*,
_class"
 loc:@pi/actor/actor/dense/bias*
valueB
 *w¾?*
dtype0*
_output_shapes
: 
©
beta2_powerVarHandleOp*
shared_namebeta2_power*,
_class"
 loc:@pi/actor/actor/dense/bias*
	container *
shape: *
dtype0*
_output_shapes
: 

,beta2_power/IsInitialized/VarIsInitializedOpVarIsInitializedOpbeta2_power*
_output_shapes
: *,
_class"
 loc:@pi/actor/actor/dense/bias

beta2_power/AssignAssignVariableOpbeta2_power%beta2_power/Initializer/initial_value*
dtype0*,
_class"
 loc:@pi/actor/actor/dense/bias

beta2_power/Read/ReadVariableOpReadVariableOpbeta2_power*,
_class"
 loc:@pi/actor/actor/dense/bias*
dtype0*
_output_shapes
: 
Ć
Bpi/actor/actor/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*.
_class$
" loc:@pi/actor/actor/dense/kernel*
valueB"      *
dtype0
­
8pi/actor/actor/dense/kernel/Adam/Initializer/zeros/ConstConst*.
_class$
" loc:@pi/actor/actor/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¤
2pi/actor/actor/dense/kernel/Adam/Initializer/zerosFillBpi/actor/actor/dense/kernel/Adam/Initializer/zeros/shape_as_tensor8pi/actor/actor/dense/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	*
T0*.
_class$
" loc:@pi/actor/actor/dense/kernel*

index_type0
Ž
 pi/actor/actor/dense/kernel/AdamVarHandleOp*
shape:	*
dtype0*
_output_shapes
: *1
shared_name" pi/actor/actor/dense/kernel/Adam*.
_class$
" loc:@pi/actor/actor/dense/kernel*
	container 
Į
Api/actor/actor/dense/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp pi/actor/actor/dense/kernel/Adam*
_output_shapes
: *.
_class$
" loc:@pi/actor/actor/dense/kernel
Ī
'pi/actor/actor/dense/kernel/Adam/AssignAssignVariableOp pi/actor/actor/dense/kernel/Adam2pi/actor/actor/dense/kernel/Adam/Initializer/zeros*.
_class$
" loc:@pi/actor/actor/dense/kernel*
dtype0
Ę
4pi/actor/actor/dense/kernel/Adam/Read/ReadVariableOpReadVariableOp pi/actor/actor/dense/kernel/Adam*
_output_shapes
:	*.
_class$
" loc:@pi/actor/actor/dense/kernel*
dtype0
Å
Dpi/actor/actor/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*.
_class$
" loc:@pi/actor/actor/dense/kernel*
valueB"      *
dtype0
Æ
:pi/actor/actor/dense/kernel/Adam_1/Initializer/zeros/ConstConst*.
_class$
" loc:@pi/actor/actor/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ŗ
4pi/actor/actor/dense/kernel/Adam_1/Initializer/zerosFillDpi/actor/actor/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor:pi/actor/actor/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*.
_class$
" loc:@pi/actor/actor/dense/kernel*

index_type0*
_output_shapes
:	
ā
"pi/actor/actor/dense/kernel/Adam_1VarHandleOp*
shape:	*
dtype0*
_output_shapes
: *3
shared_name$"pi/actor/actor/dense/kernel/Adam_1*.
_class$
" loc:@pi/actor/actor/dense/kernel*
	container 
Å
Cpi/actor/actor/dense/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp"pi/actor/actor/dense/kernel/Adam_1*.
_class$
" loc:@pi/actor/actor/dense/kernel*
_output_shapes
: 
Ō
)pi/actor/actor/dense/kernel/Adam_1/AssignAssignVariableOp"pi/actor/actor/dense/kernel/Adam_14pi/actor/actor/dense/kernel/Adam_1/Initializer/zeros*.
_class$
" loc:@pi/actor/actor/dense/kernel*
dtype0
Ź
6pi/actor/actor/dense/kernel/Adam_1/Read/ReadVariableOpReadVariableOp"pi/actor/actor/dense/kernel/Adam_1*.
_class$
" loc:@pi/actor/actor/dense/kernel*
dtype0*
_output_shapes
:	
­
0pi/actor/actor/dense/bias/Adam/Initializer/zerosConst*,
_class"
 loc:@pi/actor/actor/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ō
pi/actor/actor/dense/bias/AdamVarHandleOp*/
shared_name pi/actor/actor/dense/bias/Adam*,
_class"
 loc:@pi/actor/actor/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
: 
»
?pi/actor/actor/dense/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOppi/actor/actor/dense/bias/Adam*,
_class"
 loc:@pi/actor/actor/dense/bias*
_output_shapes
: 
Ę
%pi/actor/actor/dense/bias/Adam/AssignAssignVariableOppi/actor/actor/dense/bias/Adam0pi/actor/actor/dense/bias/Adam/Initializer/zeros*,
_class"
 loc:@pi/actor/actor/dense/bias*
dtype0
¼
2pi/actor/actor/dense/bias/Adam/Read/ReadVariableOpReadVariableOppi/actor/actor/dense/bias/Adam*,
_class"
 loc:@pi/actor/actor/dense/bias*
dtype0*
_output_shapes	
:
Æ
2pi/actor/actor/dense/bias/Adam_1/Initializer/zerosConst*,
_class"
 loc:@pi/actor/actor/dense/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ų
 pi/actor/actor/dense/bias/Adam_1VarHandleOp*1
shared_name" pi/actor/actor/dense/bias/Adam_1*,
_class"
 loc:@pi/actor/actor/dense/bias*
	container *
shape:*
dtype0*
_output_shapes
: 
æ
Api/actor/actor/dense/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp pi/actor/actor/dense/bias/Adam_1*,
_class"
 loc:@pi/actor/actor/dense/bias*
_output_shapes
: 
Ģ
'pi/actor/actor/dense/bias/Adam_1/AssignAssignVariableOp pi/actor/actor/dense/bias/Adam_12pi/actor/actor/dense/bias/Adam_1/Initializer/zeros*,
_class"
 loc:@pi/actor/actor/dense/bias*
dtype0
Ą
4pi/actor/actor/dense/bias/Adam_1/Read/ReadVariableOpReadVariableOp pi/actor/actor/dense/bias/Adam_1*,
_class"
 loc:@pi/actor/actor/dense/bias*
dtype0*
_output_shapes	
:
Ē
Dpi/actor/actor/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*0
_class&
$"loc:@pi/actor/actor/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:
±
:pi/actor/actor/dense_1/kernel/Adam/Initializer/zeros/ConstConst*0
_class&
$"loc:@pi/actor/actor/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
­
4pi/actor/actor/dense_1/kernel/Adam/Initializer/zerosFillDpi/actor/actor/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor:pi/actor/actor/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*0
_class&
$"loc:@pi/actor/actor/dense_1/kernel*

index_type0* 
_output_shapes
:

å
"pi/actor/actor/dense_1/kernel/AdamVarHandleOp*3
shared_name$"pi/actor/actor/dense_1/kernel/Adam*0
_class&
$"loc:@pi/actor/actor/dense_1/kernel*
	container *
shape:
*
dtype0*
_output_shapes
: 
Ē
Cpi/actor/actor/dense_1/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp"pi/actor/actor/dense_1/kernel/Adam*
_output_shapes
: *0
_class&
$"loc:@pi/actor/actor/dense_1/kernel
Ö
)pi/actor/actor/dense_1/kernel/Adam/AssignAssignVariableOp"pi/actor/actor/dense_1/kernel/Adam4pi/actor/actor/dense_1/kernel/Adam/Initializer/zeros*0
_class&
$"loc:@pi/actor/actor/dense_1/kernel*
dtype0
Ķ
6pi/actor/actor/dense_1/kernel/Adam/Read/ReadVariableOpReadVariableOp"pi/actor/actor/dense_1/kernel/Adam*0
_class&
$"loc:@pi/actor/actor/dense_1/kernel*
dtype0* 
_output_shapes
:

É
Fpi/actor/actor/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*0
_class&
$"loc:@pi/actor/actor/dense_1/kernel*
valueB"      *
dtype0*
_output_shapes
:
³
<pi/actor/actor/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@pi/actor/actor/dense_1/kernel*
valueB
 *    
³
6pi/actor/actor/dense_1/kernel/Adam_1/Initializer/zerosFillFpi/actor/actor/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor<pi/actor/actor/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*0
_class&
$"loc:@pi/actor/actor/dense_1/kernel*

index_type0* 
_output_shapes
:

é
$pi/actor/actor/dense_1/kernel/Adam_1VarHandleOp*
shape:
*
dtype0*
_output_shapes
: *5
shared_name&$pi/actor/actor/dense_1/kernel/Adam_1*0
_class&
$"loc:@pi/actor/actor/dense_1/kernel*
	container 
Ė
Epi/actor/actor/dense_1/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp$pi/actor/actor/dense_1/kernel/Adam_1*0
_class&
$"loc:@pi/actor/actor/dense_1/kernel*
_output_shapes
: 
Ü
+pi/actor/actor/dense_1/kernel/Adam_1/AssignAssignVariableOp$pi/actor/actor/dense_1/kernel/Adam_16pi/actor/actor/dense_1/kernel/Adam_1/Initializer/zeros*0
_class&
$"loc:@pi/actor/actor/dense_1/kernel*
dtype0
Ń
8pi/actor/actor/dense_1/kernel/Adam_1/Read/ReadVariableOpReadVariableOp$pi/actor/actor/dense_1/kernel/Adam_1*0
_class&
$"loc:@pi/actor/actor/dense_1/kernel*
dtype0* 
_output_shapes
:

±
2pi/actor/actor/dense_1/bias/Adam/Initializer/zerosConst*.
_class$
" loc:@pi/actor/actor/dense_1/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ś
 pi/actor/actor/dense_1/bias/AdamVarHandleOp*1
shared_name" pi/actor/actor/dense_1/bias/Adam*.
_class$
" loc:@pi/actor/actor/dense_1/bias*
	container *
shape:*
dtype0*
_output_shapes
: 
Į
Api/actor/actor/dense_1/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp pi/actor/actor/dense_1/bias/Adam*.
_class$
" loc:@pi/actor/actor/dense_1/bias*
_output_shapes
: 
Ī
'pi/actor/actor/dense_1/bias/Adam/AssignAssignVariableOp pi/actor/actor/dense_1/bias/Adam2pi/actor/actor/dense_1/bias/Adam/Initializer/zeros*.
_class$
" loc:@pi/actor/actor/dense_1/bias*
dtype0
Ā
4pi/actor/actor/dense_1/bias/Adam/Read/ReadVariableOpReadVariableOp pi/actor/actor/dense_1/bias/Adam*.
_class$
" loc:@pi/actor/actor/dense_1/bias*
dtype0*
_output_shapes	
:
³
4pi/actor/actor/dense_1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*.
_class$
" loc:@pi/actor/actor/dense_1/bias*
valueB*    
Ž
"pi/actor/actor/dense_1/bias/Adam_1VarHandleOp*
shape:*
dtype0*
_output_shapes
: *3
shared_name$"pi/actor/actor/dense_1/bias/Adam_1*.
_class$
" loc:@pi/actor/actor/dense_1/bias*
	container 
Å
Cpi/actor/actor/dense_1/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp"pi/actor/actor/dense_1/bias/Adam_1*
_output_shapes
: *.
_class$
" loc:@pi/actor/actor/dense_1/bias
Ō
)pi/actor/actor/dense_1/bias/Adam_1/AssignAssignVariableOp"pi/actor/actor/dense_1/bias/Adam_14pi/actor/actor/dense_1/bias/Adam_1/Initializer/zeros*.
_class$
" loc:@pi/actor/actor/dense_1/bias*
dtype0
Ę
6pi/actor/actor/dense_1/bias/Adam_1/Read/ReadVariableOpReadVariableOp"pi/actor/actor/dense_1/bias/Adam_1*.
_class$
" loc:@pi/actor/actor/dense_1/bias*
dtype0*
_output_shapes	
:
Ē
Dpi/actor/actor/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel*
valueB"      *
dtype0*
_output_shapes
:
±
:pi/actor/actor/dense_2/kernel/Adam/Initializer/zeros/ConstConst*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¬
4pi/actor/actor/dense_2/kernel/Adam/Initializer/zerosFillDpi/actor/actor/dense_2/kernel/Adam/Initializer/zeros/shape_as_tensor:pi/actor/actor/dense_2/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	*
T0*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel*

index_type0
ä
"pi/actor/actor/dense_2/kernel/AdamVarHandleOp*
	container *
shape:	*
dtype0*
_output_shapes
: *3
shared_name$"pi/actor/actor/dense_2/kernel/Adam*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel
Ē
Cpi/actor/actor/dense_2/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp"pi/actor/actor/dense_2/kernel/Adam*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel*
_output_shapes
: 
Ö
)pi/actor/actor/dense_2/kernel/Adam/AssignAssignVariableOp"pi/actor/actor/dense_2/kernel/Adam4pi/actor/actor/dense_2/kernel/Adam/Initializer/zeros*
dtype0*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel
Ģ
6pi/actor/actor/dense_2/kernel/Adam/Read/ReadVariableOpReadVariableOp"pi/actor/actor/dense_2/kernel/Adam*
_output_shapes
:	*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel*
dtype0
É
Fpi/actor/actor/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel*
valueB"      *
dtype0*
_output_shapes
:
³
<pi/actor/actor/dense_2/kernel/Adam_1/Initializer/zeros/ConstConst*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
²
6pi/actor/actor/dense_2/kernel/Adam_1/Initializer/zerosFillFpi/actor/actor/dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensor<pi/actor/actor/dense_2/kernel/Adam_1/Initializer/zeros/Const*
T0*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel*

index_type0*
_output_shapes
:	
č
$pi/actor/actor/dense_2/kernel/Adam_1VarHandleOp*
	container *
shape:	*
dtype0*
_output_shapes
: *5
shared_name&$pi/actor/actor/dense_2/kernel/Adam_1*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel
Ė
Epi/actor/actor/dense_2/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp$pi/actor/actor/dense_2/kernel/Adam_1*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel*
_output_shapes
: 
Ü
+pi/actor/actor/dense_2/kernel/Adam_1/AssignAssignVariableOp$pi/actor/actor/dense_2/kernel/Adam_16pi/actor/actor/dense_2/kernel/Adam_1/Initializer/zeros*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel*
dtype0
Š
8pi/actor/actor/dense_2/kernel/Adam_1/Read/ReadVariableOpReadVariableOp$pi/actor/actor/dense_2/kernel/Adam_1*
_output_shapes
:	*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel*
dtype0
Æ
2pi/actor/actor/dense_2/bias/Adam/Initializer/zerosConst*
_output_shapes
:*.
_class$
" loc:@pi/actor/actor/dense_2/bias*
valueB*    *
dtype0
Ł
 pi/actor/actor/dense_2/bias/AdamVarHandleOp*1
shared_name" pi/actor/actor/dense_2/bias/Adam*.
_class$
" loc:@pi/actor/actor/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
: 
Į
Api/actor/actor/dense_2/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp pi/actor/actor/dense_2/bias/Adam*
_output_shapes
: *.
_class$
" loc:@pi/actor/actor/dense_2/bias
Ī
'pi/actor/actor/dense_2/bias/Adam/AssignAssignVariableOp pi/actor/actor/dense_2/bias/Adam2pi/actor/actor/dense_2/bias/Adam/Initializer/zeros*.
_class$
" loc:@pi/actor/actor/dense_2/bias*
dtype0
Į
4pi/actor/actor/dense_2/bias/Adam/Read/ReadVariableOpReadVariableOp pi/actor/actor/dense_2/bias/Adam*.
_class$
" loc:@pi/actor/actor/dense_2/bias*
dtype0*
_output_shapes
:
±
4pi/actor/actor/dense_2/bias/Adam_1/Initializer/zerosConst*.
_class$
" loc:@pi/actor/actor/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
Ż
"pi/actor/actor/dense_2/bias/Adam_1VarHandleOp*3
shared_name$"pi/actor/actor/dense_2/bias/Adam_1*.
_class$
" loc:@pi/actor/actor/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
: 
Å
Cpi/actor/actor/dense_2/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp"pi/actor/actor/dense_2/bias/Adam_1*.
_class$
" loc:@pi/actor/actor/dense_2/bias*
_output_shapes
: 
Ō
)pi/actor/actor/dense_2/bias/Adam_1/AssignAssignVariableOp"pi/actor/actor/dense_2/bias/Adam_14pi/actor/actor/dense_2/bias/Adam_1/Initializer/zeros*.
_class$
" loc:@pi/actor/actor/dense_2/bias*
dtype0
Å
6pi/actor/actor/dense_2/bias/Adam_1/Read/ReadVariableOpReadVariableOp"pi/actor/actor/dense_2/bias/Adam_1*.
_class$
" loc:@pi/actor/actor/dense_2/bias*
dtype0*
_output_shapes
:
Ē
Dpi/actor/actor/dense_3/kernel/Adam/Initializer/zeros/shape_as_tensorConst*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*
valueB"      *
dtype0*
_output_shapes
:
±
:pi/actor/actor/dense_3/kernel/Adam/Initializer/zeros/ConstConst*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¬
4pi/actor/actor/dense_3/kernel/Adam/Initializer/zerosFillDpi/actor/actor/dense_3/kernel/Adam/Initializer/zeros/shape_as_tensor:pi/actor/actor/dense_3/kernel/Adam/Initializer/zeros/Const*
T0*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*

index_type0*
_output_shapes
:	
ä
"pi/actor/actor/dense_3/kernel/AdamVarHandleOp*3
shared_name$"pi/actor/actor/dense_3/kernel/Adam*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*
	container *
shape:	*
dtype0*
_output_shapes
: 
Ē
Cpi/actor/actor/dense_3/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp"pi/actor/actor/dense_3/kernel/Adam*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*
_output_shapes
: 
Ö
)pi/actor/actor/dense_3/kernel/Adam/AssignAssignVariableOp"pi/actor/actor/dense_3/kernel/Adam4pi/actor/actor/dense_3/kernel/Adam/Initializer/zeros*
dtype0*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel
Ģ
6pi/actor/actor/dense_3/kernel/Adam/Read/ReadVariableOpReadVariableOp"pi/actor/actor/dense_3/kernel/Adam*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*
dtype0*
_output_shapes
:	
É
Fpi/actor/actor/dense_3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*
valueB"      *
dtype0*
_output_shapes
:
³
<pi/actor/actor/dense_3/kernel/Adam_1/Initializer/zeros/ConstConst*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
²
6pi/actor/actor/dense_3/kernel/Adam_1/Initializer/zerosFillFpi/actor/actor/dense_3/kernel/Adam_1/Initializer/zeros/shape_as_tensor<pi/actor/actor/dense_3/kernel/Adam_1/Initializer/zeros/Const*
T0*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*

index_type0*
_output_shapes
:	
č
$pi/actor/actor/dense_3/kernel/Adam_1VarHandleOp*
dtype0*
_output_shapes
: *5
shared_name&$pi/actor/actor/dense_3/kernel/Adam_1*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*
	container *
shape:	
Ė
Epi/actor/actor/dense_3/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp$pi/actor/actor/dense_3/kernel/Adam_1*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*
_output_shapes
: 
Ü
+pi/actor/actor/dense_3/kernel/Adam_1/AssignAssignVariableOp$pi/actor/actor/dense_3/kernel/Adam_16pi/actor/actor/dense_3/kernel/Adam_1/Initializer/zeros*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*
dtype0
Š
8pi/actor/actor/dense_3/kernel/Adam_1/Read/ReadVariableOpReadVariableOp$pi/actor/actor/dense_3/kernel/Adam_1*
_output_shapes
:	*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*
dtype0
Æ
2pi/actor/actor/dense_3/bias/Adam/Initializer/zerosConst*.
_class$
" loc:@pi/actor/actor/dense_3/bias*
valueB*    *
dtype0*
_output_shapes
:
Ł
 pi/actor/actor/dense_3/bias/AdamVarHandleOp*1
shared_name" pi/actor/actor/dense_3/bias/Adam*.
_class$
" loc:@pi/actor/actor/dense_3/bias*
	container *
shape:*
dtype0*
_output_shapes
: 
Į
Api/actor/actor/dense_3/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp pi/actor/actor/dense_3/bias/Adam*.
_class$
" loc:@pi/actor/actor/dense_3/bias*
_output_shapes
: 
Ī
'pi/actor/actor/dense_3/bias/Adam/AssignAssignVariableOp pi/actor/actor/dense_3/bias/Adam2pi/actor/actor/dense_3/bias/Adam/Initializer/zeros*.
_class$
" loc:@pi/actor/actor/dense_3/bias*
dtype0
Į
4pi/actor/actor/dense_3/bias/Adam/Read/ReadVariableOpReadVariableOp pi/actor/actor/dense_3/bias/Adam*.
_class$
" loc:@pi/actor/actor/dense_3/bias*
dtype0*
_output_shapes
:
±
4pi/actor/actor/dense_3/bias/Adam_1/Initializer/zerosConst*.
_class$
" loc:@pi/actor/actor/dense_3/bias*
valueB*    *
dtype0*
_output_shapes
:
Ż
"pi/actor/actor/dense_3/bias/Adam_1VarHandleOp*3
shared_name$"pi/actor/actor/dense_3/bias/Adam_1*.
_class$
" loc:@pi/actor/actor/dense_3/bias*
	container *
shape:*
dtype0*
_output_shapes
: 
Å
Cpi/actor/actor/dense_3/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp"pi/actor/actor/dense_3/bias/Adam_1*
_output_shapes
: *.
_class$
" loc:@pi/actor/actor/dense_3/bias
Ō
)pi/actor/actor/dense_3/bias/Adam_1/AssignAssignVariableOp"pi/actor/actor/dense_3/bias/Adam_14pi/actor/actor/dense_3/bias/Adam_1/Initializer/zeros*
dtype0*.
_class$
" loc:@pi/actor/actor/dense_3/bias
Å
6pi/actor/actor/dense_3/bias/Adam_1/Read/ReadVariableOpReadVariableOp"pi/actor/actor/dense_3/bias/Adam_1*.
_class$
" loc:@pi/actor/actor/dense_3/bias*
dtype0*
_output_shapes
:
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w¾?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *wĢ+2

HAdam/update_pi/actor/actor/dense/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
dtype0*
_output_shapes
: 

JAdam/update_pi/actor/actor/dense/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
dtype0*
_output_shapes
: 

9Adam/update_pi/actor/actor/dense/kernel/ResourceApplyAdamResourceApplyAdampi/actor/actor/dense/kernel pi/actor/actor/dense/kernel/Adam"pi/actor/actor/dense/kernel/Adam_1HAdam/update_pi/actor/actor/dense/kernel/ResourceApplyAdam/ReadVariableOpJAdam/update_pi/actor/actor/dense/kernel/ResourceApplyAdam/ReadVariableOp_1Placeholder_10
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_0*
T0*.
_class$
" loc:@pi/actor/actor/dense/kernel*
use_nesterov( *
use_locking( 

FAdam/update_pi/actor/actor/dense/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

HAdam/update_pi/actor/actor/dense/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
dtype0*
_output_shapes
: 
ū
7Adam/update_pi/actor/actor/dense/bias/ResourceApplyAdamResourceApplyAdampi/actor/actor/dense/biaspi/actor/actor/dense/bias/Adam pi/actor/actor/dense/bias/Adam_1FAdam/update_pi/actor/actor/dense/bias/ResourceApplyAdam/ReadVariableOpHAdam/update_pi/actor/actor/dense/bias/ResourceApplyAdam/ReadVariableOp_1Placeholder_10
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_1*,
_class"
 loc:@pi/actor/actor/dense/bias*
use_nesterov( *
use_locking( *
T0

JAdam/update_pi/actor/actor/dense_1/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
dtype0*
_output_shapes
: 

LAdam/update_pi/actor/actor/dense_1/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
dtype0*
_output_shapes
: 

;Adam/update_pi/actor/actor/dense_1/kernel/ResourceApplyAdamResourceApplyAdampi/actor/actor/dense_1/kernel"pi/actor/actor/dense_1/kernel/Adam$pi/actor/actor/dense_1/kernel/Adam_1JAdam/update_pi/actor/actor/dense_1/kernel/ResourceApplyAdam/ReadVariableOpLAdam/update_pi/actor/actor/dense_1/kernel/ResourceApplyAdam/ReadVariableOp_1Placeholder_10
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_2*0
_class&
$"loc:@pi/actor/actor/dense_1/kernel*
use_nesterov( *
use_locking( *
T0

HAdam/update_pi/actor/actor/dense_1/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
dtype0*
_output_shapes
: 

JAdam/update_pi/actor/actor/dense_1/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
dtype0*
_output_shapes
: 

9Adam/update_pi/actor/actor/dense_1/bias/ResourceApplyAdamResourceApplyAdampi/actor/actor/dense_1/bias pi/actor/actor/dense_1/bias/Adam"pi/actor/actor/dense_1/bias/Adam_1HAdam/update_pi/actor/actor/dense_1/bias/ResourceApplyAdam/ReadVariableOpJAdam/update_pi/actor/actor/dense_1/bias/ResourceApplyAdam/ReadVariableOp_1Placeholder_10
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_3*
T0*.
_class$
" loc:@pi/actor/actor/dense_1/bias*
use_nesterov( *
use_locking( 

JAdam/update_pi/actor/actor/dense_2/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
_output_shapes
: *
dtype0

LAdam/update_pi/actor/actor/dense_2/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
dtype0*
_output_shapes
: 

;Adam/update_pi/actor/actor/dense_2/kernel/ResourceApplyAdamResourceApplyAdampi/actor/actor/dense_2/kernel"pi/actor/actor/dense_2/kernel/Adam$pi/actor/actor/dense_2/kernel/Adam_1JAdam/update_pi/actor/actor/dense_2/kernel/ResourceApplyAdam/ReadVariableOpLAdam/update_pi/actor/actor/dense_2/kernel/ResourceApplyAdam/ReadVariableOp_1Placeholder_10
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_4*0
_class&
$"loc:@pi/actor/actor/dense_2/kernel*
use_nesterov( *
use_locking( *
T0

HAdam/update_pi/actor/actor/dense_2/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
dtype0*
_output_shapes
: 

JAdam/update_pi/actor/actor/dense_2/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
dtype0*
_output_shapes
: 

9Adam/update_pi/actor/actor/dense_2/bias/ResourceApplyAdamResourceApplyAdampi/actor/actor/dense_2/bias pi/actor/actor/dense_2/bias/Adam"pi/actor/actor/dense_2/bias/Adam_1HAdam/update_pi/actor/actor/dense_2/bias/ResourceApplyAdam/ReadVariableOpJAdam/update_pi/actor/actor/dense_2/bias/ResourceApplyAdam/ReadVariableOp_1Placeholder_10
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_5*.
_class$
" loc:@pi/actor/actor/dense_2/bias*
use_nesterov( *
use_locking( *
T0

JAdam/update_pi/actor/actor/dense_3/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
dtype0*
_output_shapes
: 

LAdam/update_pi/actor/actor/dense_3/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
dtype0*
_output_shapes
: 

;Adam/update_pi/actor/actor/dense_3/kernel/ResourceApplyAdamResourceApplyAdampi/actor/actor/dense_3/kernel"pi/actor/actor/dense_3/kernel/Adam$pi/actor/actor/dense_3/kernel/Adam_1JAdam/update_pi/actor/actor/dense_3/kernel/ResourceApplyAdam/ReadVariableOpLAdam/update_pi/actor/actor/dense_3/kernel/ResourceApplyAdam/ReadVariableOp_1Placeholder_10
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_6*0
_class&
$"loc:@pi/actor/actor/dense_3/kernel*
use_nesterov( *
use_locking( *
T0

HAdam/update_pi/actor/actor/dense_3/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power*
dtype0*
_output_shapes
: 

JAdam/update_pi/actor/actor/dense_3/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power*
dtype0*
_output_shapes
: 

9Adam/update_pi/actor/actor/dense_3/bias/ResourceApplyAdamResourceApplyAdampi/actor/actor/dense_3/bias pi/actor/actor/dense_3/bias/Adam"pi/actor/actor/dense_3/bias/Adam_1HAdam/update_pi/actor/actor/dense_3/bias/ResourceApplyAdam/ReadVariableOpJAdam/update_pi/actor/actor/dense_3/bias/ResourceApplyAdam/ReadVariableOp_1Placeholder_10
Adam/beta1
Adam/beta2Adam/epsilon*clip_by_global_norm/clip_by_global_norm/_7*
use_locking( *
T0*.
_class$
" loc:@pi/actor/actor/dense_3/bias*
use_nesterov( 
»
Adam/ReadVariableOpReadVariableOpbeta1_power8^Adam/update_pi/actor/actor/dense/bias/ResourceApplyAdam:^Adam/update_pi/actor/actor/dense/kernel/ResourceApplyAdam:^Adam/update_pi/actor/actor/dense_1/bias/ResourceApplyAdam<^Adam/update_pi/actor/actor/dense_1/kernel/ResourceApplyAdam:^Adam/update_pi/actor/actor/dense_2/bias/ResourceApplyAdam<^Adam/update_pi/actor/actor/dense_2/kernel/ResourceApplyAdam:^Adam/update_pi/actor/actor/dense_3/bias/ResourceApplyAdam<^Adam/update_pi/actor/actor/dense_3/kernel/ResourceApplyAdam*
dtype0*
_output_shapes
: 

Adam/mulMulAdam/ReadVariableOp
Adam/beta1*
T0*,
_class"
 loc:@pi/actor/actor/dense/bias*
_output_shapes
: 
{
Adam/AssignVariableOpAssignVariableOpbeta1_powerAdam/mul*,
_class"
 loc:@pi/actor/actor/dense/bias*
dtype0

Adam/ReadVariableOp_1ReadVariableOpbeta1_power^Adam/AssignVariableOp8^Adam/update_pi/actor/actor/dense/bias/ResourceApplyAdam:^Adam/update_pi/actor/actor/dense/kernel/ResourceApplyAdam:^Adam/update_pi/actor/actor/dense_1/bias/ResourceApplyAdam<^Adam/update_pi/actor/actor/dense_1/kernel/ResourceApplyAdam:^Adam/update_pi/actor/actor/dense_2/bias/ResourceApplyAdam<^Adam/update_pi/actor/actor/dense_2/kernel/ResourceApplyAdam:^Adam/update_pi/actor/actor/dense_3/bias/ResourceApplyAdam<^Adam/update_pi/actor/actor/dense_3/kernel/ResourceApplyAdam*,
_class"
 loc:@pi/actor/actor/dense/bias*
dtype0*
_output_shapes
: 
½
Adam/ReadVariableOp_2ReadVariableOpbeta2_power8^Adam/update_pi/actor/actor/dense/bias/ResourceApplyAdam:^Adam/update_pi/actor/actor/dense/kernel/ResourceApplyAdam:^Adam/update_pi/actor/actor/dense_1/bias/ResourceApplyAdam<^Adam/update_pi/actor/actor/dense_1/kernel/ResourceApplyAdam:^Adam/update_pi/actor/actor/dense_2/bias/ResourceApplyAdam<^Adam/update_pi/actor/actor/dense_2/kernel/ResourceApplyAdam:^Adam/update_pi/actor/actor/dense_3/bias/ResourceApplyAdam<^Adam/update_pi/actor/actor/dense_3/kernel/ResourceApplyAdam*
dtype0*
_output_shapes
: 


Adam/mul_1MulAdam/ReadVariableOp_2
Adam/beta2*
T0*,
_class"
 loc:@pi/actor/actor/dense/bias*
_output_shapes
: 

Adam/AssignVariableOp_1AssignVariableOpbeta2_power
Adam/mul_1*
dtype0*,
_class"
 loc:@pi/actor/actor/dense/bias

Adam/ReadVariableOp_3ReadVariableOpbeta2_power^Adam/AssignVariableOp_18^Adam/update_pi/actor/actor/dense/bias/ResourceApplyAdam:^Adam/update_pi/actor/actor/dense/kernel/ResourceApplyAdam:^Adam/update_pi/actor/actor/dense_1/bias/ResourceApplyAdam<^Adam/update_pi/actor/actor/dense_1/kernel/ResourceApplyAdam:^Adam/update_pi/actor/actor/dense_2/bias/ResourceApplyAdam<^Adam/update_pi/actor/actor/dense_2/kernel/ResourceApplyAdam:^Adam/update_pi/actor/actor/dense_3/bias/ResourceApplyAdam<^Adam/update_pi/actor/actor/dense_3/kernel/ResourceApplyAdam*,
_class"
 loc:@pi/actor/actor/dense/bias*
dtype0*
_output_shapes
: 
¢
AdamNoOp^Adam/AssignVariableOp^Adam/AssignVariableOp_18^Adam/update_pi/actor/actor/dense/bias/ResourceApplyAdam:^Adam/update_pi/actor/actor/dense/kernel/ResourceApplyAdam:^Adam/update_pi/actor/actor/dense_1/bias/ResourceApplyAdam<^Adam/update_pi/actor/actor/dense_1/kernel/ResourceApplyAdam:^Adam/update_pi/actor/actor/dense_2/bias/ResourceApplyAdam<^Adam/update_pi/actor/actor/dense_2/kernel/ResourceApplyAdam:^Adam/update_pi/actor/actor/dense_3/bias/ResourceApplyAdam<^Adam/update_pi/actor/actor/dense_3/kernel/ResourceApplyAdam
T
gradients_1/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_1/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
_output_shapes
: *
T0*

index_type0
o
%gradients_1/Mean_1_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

gradients_1/Mean_1_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
b
gradients_1/Mean_1_grad/ShapeShapepow_2*
T0*
out_type0*
_output_shapes
:
¤
gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*#
_output_shapes
:’’’’’’’’’*

Tmultiples0*
T0
d
gradients_1/Mean_1_grad/Shape_1Shapepow_2*
T0*
out_type0*
_output_shapes
:
b
gradients_1/Mean_1_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_1/Mean_1_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
¢
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
i
gradients_1/Mean_1_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
¦
gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
c
!gradients_1/Mean_1_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :

gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
_output_shapes
: *
T0

 gradients_1/Mean_1_grad/floordivFloorDivgradients_1/Mean_1_grad/Prodgradients_1/Mean_1_grad/Maximum*
_output_shapes
: *
T0

gradients_1/Mean_1_grad/CastCast gradients_1/Mean_1_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0

gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*
T0*#
_output_shapes
:’’’’’’’’’
a
gradients_1/pow_2_grad/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
a
gradients_1/pow_2_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ą
,gradients_1/pow_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pow_2_grad/Shapegradients_1/pow_2_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
y
gradients_1/pow_2_grad/mulMulgradients_1/Mean_1_grad/truedivpow_2/y*#
_output_shapes
:’’’’’’’’’*
T0
a
gradients_1/pow_2_grad/sub/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
i
gradients_1/pow_2_grad/subSubpow_2/ygradients_1/pow_2_grad/sub/y*
_output_shapes
: *
T0
r
gradients_1/pow_2_grad/PowPowsub_1gradients_1/pow_2_grad/sub*#
_output_shapes
:’’’’’’’’’*
T0

gradients_1/pow_2_grad/mul_1Mulgradients_1/pow_2_grad/mulgradients_1/pow_2_grad/Pow*
T0*#
_output_shapes
:’’’’’’’’’
­
gradients_1/pow_2_grad/SumSumgradients_1/pow_2_grad/mul_1,gradients_1/pow_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0

gradients_1/pow_2_grad/ReshapeReshapegradients_1/pow_2_grad/Sumgradients_1/pow_2_grad/Shape*#
_output_shapes
:’’’’’’’’’*
T0*
Tshape0
e
 gradients_1/pow_2_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients_1/pow_2_grad/GreaterGreatersub_1 gradients_1/pow_2_grad/Greater/y*#
_output_shapes
:’’’’’’’’’*
T0
k
&gradients_1/pow_2_grad/ones_like/ShapeShapesub_1*
out_type0*
_output_shapes
:*
T0
k
&gradients_1/pow_2_grad/ones_like/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
ø
 gradients_1/pow_2_grad/ones_likeFill&gradients_1/pow_2_grad/ones_like/Shape&gradients_1/pow_2_grad/ones_like/Const*
T0*

index_type0*#
_output_shapes
:’’’’’’’’’

gradients_1/pow_2_grad/SelectSelectgradients_1/pow_2_grad/Greatersub_1 gradients_1/pow_2_grad/ones_like*#
_output_shapes
:’’’’’’’’’*
T0
n
gradients_1/pow_2_grad/LogLoggradients_1/pow_2_grad/Select*
T0*#
_output_shapes
:’’’’’’’’’
c
!gradients_1/pow_2_grad/zeros_like	ZerosLikesub_1*
T0*#
_output_shapes
:’’’’’’’’’
¶
gradients_1/pow_2_grad/Select_1Selectgradients_1/pow_2_grad/Greatergradients_1/pow_2_grad/Log!gradients_1/pow_2_grad/zeros_like*
T0*#
_output_shapes
:’’’’’’’’’
y
gradients_1/pow_2_grad/mul_2Mulgradients_1/Mean_1_grad/truedivpow_2*
T0*#
_output_shapes
:’’’’’’’’’

gradients_1/pow_2_grad/mul_3Mulgradients_1/pow_2_grad/mul_2gradients_1/pow_2_grad/Select_1*
T0*#
_output_shapes
:’’’’’’’’’
±
gradients_1/pow_2_grad/Sum_1Sumgradients_1/pow_2_grad/mul_3.gradients_1/pow_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

 gradients_1/pow_2_grad/Reshape_1Reshapegradients_1/pow_2_grad/Sum_1gradients_1/pow_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
s
'gradients_1/pow_2_grad/tuple/group_depsNoOp^gradients_1/pow_2_grad/Reshape!^gradients_1/pow_2_grad/Reshape_1
ę
/gradients_1/pow_2_grad/tuple/control_dependencyIdentitygradients_1/pow_2_grad/Reshape(^gradients_1/pow_2_grad/tuple/group_deps*#
_output_shapes
:’’’’’’’’’*
T0*1
_class'
%#loc:@gradients_1/pow_2_grad/Reshape
ß
1gradients_1/pow_2_grad/tuple/control_dependency_1Identity gradients_1/pow_2_grad/Reshape_1(^gradients_1/pow_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/pow_2_grad/Reshape_1*
_output_shapes
: 
i
gradients_1/sub_1_grad/ShapeShapePlaceholder_6*
T0*
out_type0*
_output_shapes
:
g
gradients_1/sub_1_grad/Shape_1Shape	v/Squeeze*
T0*
out_type0*
_output_shapes
:
Ą
,gradients_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_1_grad/Shapegradients_1/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
Ą
gradients_1/sub_1_grad/SumSum/gradients_1/pow_2_grad/tuple/control_dependency,gradients_1/sub_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients_1/sub_1_grad/ReshapeReshapegradients_1/sub_1_grad/Sumgradients_1/sub_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:’’’’’’’’’
Ä
gradients_1/sub_1_grad/Sum_1Sum/gradients_1/pow_2_grad/tuple/control_dependency.gradients_1/sub_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
b
gradients_1/sub_1_grad/NegNeggradients_1/sub_1_grad/Sum_1*
_output_shapes
:*
T0
£
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Neggradients_1/sub_1_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:’’’’’’’’’
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
ę
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*#
_output_shapes
:’’’’’’’’’*
T0*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape
ģ
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*#
_output_shapes
:’’’’’’’’’*
T0*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1

 gradients_1/v/Squeeze_grad/ShapeShapev/critic/critic/dense_6/BiasAdd*
T0*
out_type0*
_output_shapes
:
Ā
"gradients_1/v/Squeeze_grad/ReshapeReshape1gradients_1/sub_1_grad/tuple/control_dependency_1 gradients_1/v/Squeeze_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’
«
<gradients_1/v/critic/critic/dense_6/BiasAdd_grad/BiasAddGradBiasAddGrad"gradients_1/v/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:
­
Agradients_1/v/critic/critic/dense_6/BiasAdd_grad/tuple/group_depsNoOp#^gradients_1/v/Squeeze_grad/Reshape=^gradients_1/v/critic/critic/dense_6/BiasAdd_grad/BiasAddGrad
¦
Igradients_1/v/critic/critic/dense_6/BiasAdd_grad/tuple/control_dependencyIdentity"gradients_1/v/Squeeze_grad/ReshapeB^gradients_1/v/critic/critic/dense_6/BiasAdd_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/v/Squeeze_grad/Reshape*'
_output_shapes
:’’’’’’’’’
Ļ
Kgradients_1/v/critic/critic/dense_6/BiasAdd_grad/tuple/control_dependency_1Identity<gradients_1/v/critic/critic/dense_6/BiasAdd_grad/BiasAddGradB^gradients_1/v/critic/critic/dense_6/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*O
_classE
CAloc:@gradients_1/v/critic/critic/dense_6/BiasAdd_grad/BiasAddGrad

6gradients_1/v/critic/critic/dense_6/MatMul_grad/MatMulMatMulIgradients_1/v/critic/critic/dense_6/BiasAdd_grad/tuple/control_dependency-v/critic/critic/dense_6/MatMul/ReadVariableOp*
transpose_b(*
T0*(
_output_shapes
:’’’’’’’’’*
transpose_a( 

8gradients_1/v/critic/critic/dense_6/MatMul_grad/MatMul_1MatMul'v/critic/critic/leaky_re_lu_3/LeakyReluIgradients_1/v/critic/critic/dense_6/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	*
transpose_a(
¼
@gradients_1/v/critic/critic/dense_6/MatMul_grad/tuple/group_depsNoOp7^gradients_1/v/critic/critic/dense_6/MatMul_grad/MatMul9^gradients_1/v/critic/critic/dense_6/MatMul_grad/MatMul_1
Ķ
Hgradients_1/v/critic/critic/dense_6/MatMul_grad/tuple/control_dependencyIdentity6gradients_1/v/critic/critic/dense_6/MatMul_grad/MatMulA^gradients_1/v/critic/critic/dense_6/MatMul_grad/tuple/group_deps*I
_class?
=;loc:@gradients_1/v/critic/critic/dense_6/MatMul_grad/MatMul*(
_output_shapes
:’’’’’’’’’*
T0
Ź
Jgradients_1/v/critic/critic/dense_6/MatMul_grad/tuple/control_dependency_1Identity8gradients_1/v/critic/critic/dense_6/MatMul_grad/MatMul_1A^gradients_1/v/critic/critic/dense_6/MatMul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/v/critic/critic/dense_6/MatMul_grad/MatMul_1*
_output_shapes
:	

Fgradients_1/v/critic/critic/leaky_re_lu_3/LeakyRelu_grad/LeakyReluGradLeakyReluGradHgradients_1/v/critic/critic/dense_6/MatMul_grad/tuple/control_dependencyv/critic/critic/dense_5/BiasAdd*
T0*
alpha%>*(
_output_shapes
:’’’’’’’’’
Š
<gradients_1/v/critic/critic/dense_5/BiasAdd_grad/BiasAddGradBiasAddGradFgradients_1/v/critic/critic/leaky_re_lu_3/LeakyRelu_grad/LeakyReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Ń
Agradients_1/v/critic/critic/dense_5/BiasAdd_grad/tuple/group_depsNoOp=^gradients_1/v/critic/critic/dense_5/BiasAdd_grad/BiasAddGradG^gradients_1/v/critic/critic/leaky_re_lu_3/LeakyRelu_grad/LeakyReluGrad
ļ
Igradients_1/v/critic/critic/dense_5/BiasAdd_grad/tuple/control_dependencyIdentityFgradients_1/v/critic/critic/leaky_re_lu_3/LeakyRelu_grad/LeakyReluGradB^gradients_1/v/critic/critic/dense_5/BiasAdd_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients_1/v/critic/critic/leaky_re_lu_3/LeakyRelu_grad/LeakyReluGrad*(
_output_shapes
:’’’’’’’’’
Š
Kgradients_1/v/critic/critic/dense_5/BiasAdd_grad/tuple/control_dependency_1Identity<gradients_1/v/critic/critic/dense_5/BiasAdd_grad/BiasAddGradB^gradients_1/v/critic/critic/dense_5/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:*
T0*O
_classE
CAloc:@gradients_1/v/critic/critic/dense_5/BiasAdd_grad/BiasAddGrad

6gradients_1/v/critic/critic/dense_5/MatMul_grad/MatMulMatMulIgradients_1/v/critic/critic/dense_5/BiasAdd_grad/tuple/control_dependency-v/critic/critic/dense_5/MatMul/ReadVariableOp*
transpose_b(*
T0*(
_output_shapes
:’’’’’’’’’*
transpose_a( 

8gradients_1/v/critic/critic/dense_5/MatMul_grad/MatMul_1MatMul'v/critic/critic/leaky_re_lu_2/LeakyReluIgradients_1/v/critic/critic/dense_5/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(
¼
@gradients_1/v/critic/critic/dense_5/MatMul_grad/tuple/group_depsNoOp7^gradients_1/v/critic/critic/dense_5/MatMul_grad/MatMul9^gradients_1/v/critic/critic/dense_5/MatMul_grad/MatMul_1
Ķ
Hgradients_1/v/critic/critic/dense_5/MatMul_grad/tuple/control_dependencyIdentity6gradients_1/v/critic/critic/dense_5/MatMul_grad/MatMulA^gradients_1/v/critic/critic/dense_5/MatMul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients_1/v/critic/critic/dense_5/MatMul_grad/MatMul*(
_output_shapes
:’’’’’’’’’
Ė
Jgradients_1/v/critic/critic/dense_5/MatMul_grad/tuple/control_dependency_1Identity8gradients_1/v/critic/critic/dense_5/MatMul_grad/MatMul_1A^gradients_1/v/critic/critic/dense_5/MatMul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/v/critic/critic/dense_5/MatMul_grad/MatMul_1* 
_output_shapes
:


Fgradients_1/v/critic/critic/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGradLeakyReluGradHgradients_1/v/critic/critic/dense_5/MatMul_grad/tuple/control_dependencyv/critic/critic/dense_4/BiasAdd*
T0*
alpha%>*(
_output_shapes
:’’’’’’’’’
Š
<gradients_1/v/critic/critic/dense_4/BiasAdd_grad/BiasAddGradBiasAddGradFgradients_1/v/critic/critic/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad*
_output_shapes	
:*
T0*
data_formatNHWC
Ń
Agradients_1/v/critic/critic/dense_4/BiasAdd_grad/tuple/group_depsNoOp=^gradients_1/v/critic/critic/dense_4/BiasAdd_grad/BiasAddGradG^gradients_1/v/critic/critic/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad
ļ
Igradients_1/v/critic/critic/dense_4/BiasAdd_grad/tuple/control_dependencyIdentityFgradients_1/v/critic/critic/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGradB^gradients_1/v/critic/critic/dense_4/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:’’’’’’’’’*
T0*Y
_classO
MKloc:@gradients_1/v/critic/critic/leaky_re_lu_2/LeakyRelu_grad/LeakyReluGrad
Š
Kgradients_1/v/critic/critic/dense_4/BiasAdd_grad/tuple/control_dependency_1Identity<gradients_1/v/critic/critic/dense_4/BiasAdd_grad/BiasAddGradB^gradients_1/v/critic/critic/dense_4/BiasAdd_grad/tuple/group_deps*O
_classE
CAloc:@gradients_1/v/critic/critic/dense_4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0

6gradients_1/v/critic/critic/dense_4/MatMul_grad/MatMulMatMulIgradients_1/v/critic/critic/dense_4/BiasAdd_grad/tuple/control_dependency-v/critic/critic/dense_4/MatMul/ReadVariableOp*
T0*'
_output_shapes
:’’’’’’’’’*
transpose_a( *
transpose_b(
ź
8gradients_1/v/critic/critic/dense_4/MatMul_grad/MatMul_1MatMulPlaceholderIgradients_1/v/critic/critic/dense_4/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	*
transpose_a(*
transpose_b( 
¼
@gradients_1/v/critic/critic/dense_4/MatMul_grad/tuple/group_depsNoOp7^gradients_1/v/critic/critic/dense_4/MatMul_grad/MatMul9^gradients_1/v/critic/critic/dense_4/MatMul_grad/MatMul_1
Ģ
Hgradients_1/v/critic/critic/dense_4/MatMul_grad/tuple/control_dependencyIdentity6gradients_1/v/critic/critic/dense_4/MatMul_grad/MatMulA^gradients_1/v/critic/critic/dense_4/MatMul_grad/tuple/group_deps*'
_output_shapes
:’’’’’’’’’*
T0*I
_class?
=;loc:@gradients_1/v/critic/critic/dense_4/MatMul_grad/MatMul
Ź
Jgradients_1/v/critic/critic/dense_4/MatMul_grad/tuple/control_dependency_1Identity8gradients_1/v/critic/critic/dense_4/MatMul_grad/MatMul_1A^gradients_1/v/critic/critic/dense_4/MatMul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/v/critic/critic/dense_4/MatMul_grad/MatMul_1*
_output_shapes
:	
Ų
global_norm_1/L2LossL2LossJgradients_1/v/critic/critic/dense_4/MatMul_grad/tuple/control_dependency_1*
_output_shapes
: *
T0*K
_classA
?=loc:@gradients_1/v/critic/critic/dense_4/MatMul_grad/MatMul_1
ß
global_norm_1/L2Loss_1L2LossKgradients_1/v/critic/critic/dense_4/BiasAdd_grad/tuple/control_dependency_1*
T0*O
_classE
CAloc:@gradients_1/v/critic/critic/dense_4/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
Ś
global_norm_1/L2Loss_2L2LossJgradients_1/v/critic/critic/dense_5/MatMul_grad/tuple/control_dependency_1*
T0*K
_classA
?=loc:@gradients_1/v/critic/critic/dense_5/MatMul_grad/MatMul_1*
_output_shapes
: 
ß
global_norm_1/L2Loss_3L2LossKgradients_1/v/critic/critic/dense_5/BiasAdd_grad/tuple/control_dependency_1*
T0*O
_classE
CAloc:@gradients_1/v/critic/critic/dense_5/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
Ś
global_norm_1/L2Loss_4L2LossJgradients_1/v/critic/critic/dense_6/MatMul_grad/tuple/control_dependency_1*
_output_shapes
: *
T0*K
_classA
?=loc:@gradients_1/v/critic/critic/dense_6/MatMul_grad/MatMul_1
ß
global_norm_1/L2Loss_5L2LossKgradients_1/v/critic/critic/dense_6/BiasAdd_grad/tuple/control_dependency_1*
T0*O
_classE
CAloc:@gradients_1/v/critic/critic/dense_6/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
ć
global_norm_1/stackPackglobal_norm_1/L2Lossglobal_norm_1/L2Loss_1global_norm_1/L2Loss_2global_norm_1/L2Loss_3global_norm_1/L2Loss_4global_norm_1/L2Loss_5*

axis *
N*
_output_shapes
:*
T0
]
global_norm_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:

global_norm_1/SumSumglobal_norm_1/stackglobal_norm_1/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Z
global_norm_1/Const_1Const*
valueB
 *   @*
dtype0*
_output_shapes
: 
c
global_norm_1/mulMulglobal_norm_1/Sumglobal_norm_1/Const_1*
T0*
_output_shapes
: 
U
global_norm_1/global_normSqrtglobal_norm_1/mul*
T0*
_output_shapes
: 
d
clip_by_global_norm_1/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

clip_by_global_norm_1/truedivRealDivclip_by_global_norm_1/truediv/xglobal_norm_1/global_norm*
_output_shapes
: *
T0
`
clip_by_global_norm_1/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
!clip_by_global_norm_1/truediv_1/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 

clip_by_global_norm_1/truediv_1RealDivclip_by_global_norm_1/Const!clip_by_global_norm_1/truediv_1/y*
T0*
_output_shapes
: 

clip_by_global_norm_1/MinimumMinimumclip_by_global_norm_1/truedivclip_by_global_norm_1/truediv_1*
_output_shapes
: *
T0
`
clip_by_global_norm_1/mul/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
}
clip_by_global_norm_1/mulMulclip_by_global_norm_1/mul/xclip_by_global_norm_1/Minimum*
_output_shapes
: *
T0
f
clip_by_global_norm_1/IsFiniteIsFiniteglobal_norm_1/global_norm*
_output_shapes
: *
T0
b
clip_by_global_norm_1/Const_1Const*
valueB
 *  Ą*
dtype0*
_output_shapes
: 
”
clip_by_global_norm_1/SelectSelectclip_by_global_norm_1/IsFiniteclip_by_global_norm_1/mulclip_by_global_norm_1/Const_1*
T0*
_output_shapes
: 

clip_by_global_norm_1/mul_1MulJgradients_1/v/critic/critic/dense_4/MatMul_grad/tuple/control_dependency_1clip_by_global_norm_1/Select*
T0*K
_classA
?=loc:@gradients_1/v/critic/critic/dense_4/MatMul_grad/MatMul_1*
_output_shapes
:	
Ī
.clip_by_global_norm_1/clip_by_global_norm_1/_0Identityclip_by_global_norm_1/mul_1*
T0*K
_classA
?=loc:@gradients_1/v/critic/critic/dense_4/MatMul_grad/MatMul_1*
_output_shapes
:	

clip_by_global_norm_1/mul_2MulKgradients_1/v/critic/critic/dense_4/BiasAdd_grad/tuple/control_dependency_1clip_by_global_norm_1/Select*
T0*O
_classE
CAloc:@gradients_1/v/critic/critic/dense_4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ī
.clip_by_global_norm_1/clip_by_global_norm_1/_1Identityclip_by_global_norm_1/mul_2*
T0*O
_classE
CAloc:@gradients_1/v/critic/critic/dense_4/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:

clip_by_global_norm_1/mul_3MulJgradients_1/v/critic/critic/dense_5/MatMul_grad/tuple/control_dependency_1clip_by_global_norm_1/Select*
T0*K
_classA
?=loc:@gradients_1/v/critic/critic/dense_5/MatMul_grad/MatMul_1* 
_output_shapes
:

Ļ
.clip_by_global_norm_1/clip_by_global_norm_1/_2Identityclip_by_global_norm_1/mul_3* 
_output_shapes
:
*
T0*K
_classA
?=loc:@gradients_1/v/critic/critic/dense_5/MatMul_grad/MatMul_1

clip_by_global_norm_1/mul_4MulKgradients_1/v/critic/critic/dense_5/BiasAdd_grad/tuple/control_dependency_1clip_by_global_norm_1/Select*
T0*O
_classE
CAloc:@gradients_1/v/critic/critic/dense_5/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ī
.clip_by_global_norm_1/clip_by_global_norm_1/_3Identityclip_by_global_norm_1/mul_4*
_output_shapes	
:*
T0*O
_classE
CAloc:@gradients_1/v/critic/critic/dense_5/BiasAdd_grad/BiasAddGrad

clip_by_global_norm_1/mul_5MulJgradients_1/v/critic/critic/dense_6/MatMul_grad/tuple/control_dependency_1clip_by_global_norm_1/Select*
_output_shapes
:	*
T0*K
_classA
?=loc:@gradients_1/v/critic/critic/dense_6/MatMul_grad/MatMul_1
Ī
.clip_by_global_norm_1/clip_by_global_norm_1/_4Identityclip_by_global_norm_1/mul_5*
_output_shapes
:	*
T0*K
_classA
?=loc:@gradients_1/v/critic/critic/dense_6/MatMul_grad/MatMul_1

clip_by_global_norm_1/mul_6MulKgradients_1/v/critic/critic/dense_6/BiasAdd_grad/tuple/control_dependency_1clip_by_global_norm_1/Select*
T0*O
_classE
CAloc:@gradients_1/v/critic/critic/dense_6/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Ķ
.clip_by_global_norm_1/clip_by_global_norm_1/_5Identityclip_by_global_norm_1/mul_6*
_output_shapes
:*
T0*O
_classE
CAloc:@gradients_1/v/critic/critic/dense_6/BiasAdd_grad/BiasAddGrad

'beta1_power_1/Initializer/initial_valueConst*/
_class%
#!loc:@v/critic/critic/dense_4/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
°
beta1_power_1VarHandleOp*
dtype0*
_output_shapes
: *
shared_namebeta1_power_1*/
_class%
#!loc:@v/critic/critic/dense_4/bias*
	container *
shape: 

.beta1_power_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpbeta1_power_1*/
_class%
#!loc:@v/critic/critic/dense_4/bias*
_output_shapes
: 

beta1_power_1/AssignAssignVariableOpbeta1_power_1'beta1_power_1/Initializer/initial_value*/
_class%
#!loc:@v/critic/critic/dense_4/bias*
dtype0

!beta1_power_1/Read/ReadVariableOpReadVariableOpbeta1_power_1*
_output_shapes
: */
_class%
#!loc:@v/critic/critic/dense_4/bias*
dtype0

'beta2_power_1/Initializer/initial_valueConst*
_output_shapes
: */
_class%
#!loc:@v/critic/critic/dense_4/bias*
valueB
 *w¾?*
dtype0
°
beta2_power_1VarHandleOp*
_output_shapes
: *
shared_namebeta2_power_1*/
_class%
#!loc:@v/critic/critic/dense_4/bias*
	container *
shape: *
dtype0

.beta2_power_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpbeta2_power_1*/
_class%
#!loc:@v/critic/critic/dense_4/bias*
_output_shapes
: 

beta2_power_1/AssignAssignVariableOpbeta2_power_1'beta2_power_1/Initializer/initial_value*/
_class%
#!loc:@v/critic/critic/dense_4/bias*
dtype0

!beta2_power_1/Read/ReadVariableOpReadVariableOpbeta2_power_1*/
_class%
#!loc:@v/critic/critic/dense_4/bias*
dtype0*
_output_shapes
: 
É
Ev/critic/critic/dense_4/kernel/Adam/Initializer/zeros/shape_as_tensorConst*1
_class'
%#loc:@v/critic/critic/dense_4/kernel*
valueB"      *
dtype0*
_output_shapes
:
³
;v/critic/critic/dense_4/kernel/Adam/Initializer/zeros/ConstConst*1
_class'
%#loc:@v/critic/critic/dense_4/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
°
5v/critic/critic/dense_4/kernel/Adam/Initializer/zerosFillEv/critic/critic/dense_4/kernel/Adam/Initializer/zeros/shape_as_tensor;v/critic/critic/dense_4/kernel/Adam/Initializer/zeros/Const*
T0*1
_class'
%#loc:@v/critic/critic/dense_4/kernel*

index_type0*
_output_shapes
:	
ē
#v/critic/critic/dense_4/kernel/AdamVarHandleOp*
shape:	*
dtype0*
_output_shapes
: *4
shared_name%#v/critic/critic/dense_4/kernel/Adam*1
_class'
%#loc:@v/critic/critic/dense_4/kernel*
	container 
Ź
Dv/critic/critic/dense_4/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp#v/critic/critic/dense_4/kernel/Adam*
_output_shapes
: *1
_class'
%#loc:@v/critic/critic/dense_4/kernel
Ś
*v/critic/critic/dense_4/kernel/Adam/AssignAssignVariableOp#v/critic/critic/dense_4/kernel/Adam5v/critic/critic/dense_4/kernel/Adam/Initializer/zeros*
dtype0*1
_class'
%#loc:@v/critic/critic/dense_4/kernel
Ļ
7v/critic/critic/dense_4/kernel/Adam/Read/ReadVariableOpReadVariableOp#v/critic/critic/dense_4/kernel/Adam*1
_class'
%#loc:@v/critic/critic/dense_4/kernel*
dtype0*
_output_shapes
:	
Ė
Gv/critic/critic/dense_4/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*1
_class'
%#loc:@v/critic/critic/dense_4/kernel*
valueB"      *
dtype0*
_output_shapes
:
µ
=v/critic/critic/dense_4/kernel/Adam_1/Initializer/zeros/ConstConst*1
_class'
%#loc:@v/critic/critic/dense_4/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
¶
7v/critic/critic/dense_4/kernel/Adam_1/Initializer/zerosFillGv/critic/critic/dense_4/kernel/Adam_1/Initializer/zeros/shape_as_tensor=v/critic/critic/dense_4/kernel/Adam_1/Initializer/zeros/Const*1
_class'
%#loc:@v/critic/critic/dense_4/kernel*

index_type0*
_output_shapes
:	*
T0
ė
%v/critic/critic/dense_4/kernel/Adam_1VarHandleOp*6
shared_name'%v/critic/critic/dense_4/kernel/Adam_1*1
_class'
%#loc:@v/critic/critic/dense_4/kernel*
	container *
shape:	*
dtype0*
_output_shapes
: 
Ī
Fv/critic/critic/dense_4/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp%v/critic/critic/dense_4/kernel/Adam_1*
_output_shapes
: *1
_class'
%#loc:@v/critic/critic/dense_4/kernel
ą
,v/critic/critic/dense_4/kernel/Adam_1/AssignAssignVariableOp%v/critic/critic/dense_4/kernel/Adam_17v/critic/critic/dense_4/kernel/Adam_1/Initializer/zeros*1
_class'
%#loc:@v/critic/critic/dense_4/kernel*
dtype0
Ó
9v/critic/critic/dense_4/kernel/Adam_1/Read/ReadVariableOpReadVariableOp%v/critic/critic/dense_4/kernel/Adam_1*1
_class'
%#loc:@v/critic/critic/dense_4/kernel*
dtype0*
_output_shapes
:	
³
3v/critic/critic/dense_4/bias/Adam/Initializer/zerosConst*
_output_shapes	
:*/
_class%
#!loc:@v/critic/critic/dense_4/bias*
valueB*    *
dtype0
Ż
!v/critic/critic/dense_4/bias/AdamVarHandleOp*
shape:*
dtype0*
_output_shapes
: *2
shared_name#!v/critic/critic/dense_4/bias/Adam*/
_class%
#!loc:@v/critic/critic/dense_4/bias*
	container 
Ä
Bv/critic/critic/dense_4/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp!v/critic/critic/dense_4/bias/Adam*
_output_shapes
: */
_class%
#!loc:@v/critic/critic/dense_4/bias
Ņ
(v/critic/critic/dense_4/bias/Adam/AssignAssignVariableOp!v/critic/critic/dense_4/bias/Adam3v/critic/critic/dense_4/bias/Adam/Initializer/zeros*/
_class%
#!loc:@v/critic/critic/dense_4/bias*
dtype0
Å
5v/critic/critic/dense_4/bias/Adam/Read/ReadVariableOpReadVariableOp!v/critic/critic/dense_4/bias/Adam*
dtype0*
_output_shapes	
:*/
_class%
#!loc:@v/critic/critic/dense_4/bias
µ
5v/critic/critic/dense_4/bias/Adam_1/Initializer/zerosConst*/
_class%
#!loc:@v/critic/critic/dense_4/bias*
valueB*    *
dtype0*
_output_shapes	
:
į
#v/critic/critic/dense_4/bias/Adam_1VarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: *4
shared_name%#v/critic/critic/dense_4/bias/Adam_1*/
_class%
#!loc:@v/critic/critic/dense_4/bias
Č
Dv/critic/critic/dense_4/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp#v/critic/critic/dense_4/bias/Adam_1*/
_class%
#!loc:@v/critic/critic/dense_4/bias*
_output_shapes
: 
Ų
*v/critic/critic/dense_4/bias/Adam_1/AssignAssignVariableOp#v/critic/critic/dense_4/bias/Adam_15v/critic/critic/dense_4/bias/Adam_1/Initializer/zeros*/
_class%
#!loc:@v/critic/critic/dense_4/bias*
dtype0
É
7v/critic/critic/dense_4/bias/Adam_1/Read/ReadVariableOpReadVariableOp#v/critic/critic/dense_4/bias/Adam_1*
_output_shapes	
:*/
_class%
#!loc:@v/critic/critic/dense_4/bias*
dtype0
É
Ev/critic/critic/dense_5/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*1
_class'
%#loc:@v/critic/critic/dense_5/kernel*
valueB"      *
dtype0
³
;v/critic/critic/dense_5/kernel/Adam/Initializer/zeros/ConstConst*1
_class'
%#loc:@v/critic/critic/dense_5/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
±
5v/critic/critic/dense_5/kernel/Adam/Initializer/zerosFillEv/critic/critic/dense_5/kernel/Adam/Initializer/zeros/shape_as_tensor;v/critic/critic/dense_5/kernel/Adam/Initializer/zeros/Const*1
_class'
%#loc:@v/critic/critic/dense_5/kernel*

index_type0* 
_output_shapes
:
*
T0
č
#v/critic/critic/dense_5/kernel/AdamVarHandleOp*4
shared_name%#v/critic/critic/dense_5/kernel/Adam*1
_class'
%#loc:@v/critic/critic/dense_5/kernel*
	container *
shape:
*
dtype0*
_output_shapes
: 
Ź
Dv/critic/critic/dense_5/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp#v/critic/critic/dense_5/kernel/Adam*1
_class'
%#loc:@v/critic/critic/dense_5/kernel*
_output_shapes
: 
Ś
*v/critic/critic/dense_5/kernel/Adam/AssignAssignVariableOp#v/critic/critic/dense_5/kernel/Adam5v/critic/critic/dense_5/kernel/Adam/Initializer/zeros*1
_class'
%#loc:@v/critic/critic/dense_5/kernel*
dtype0
Š
7v/critic/critic/dense_5/kernel/Adam/Read/ReadVariableOpReadVariableOp#v/critic/critic/dense_5/kernel/Adam*1
_class'
%#loc:@v/critic/critic/dense_5/kernel*
dtype0* 
_output_shapes
:

Ė
Gv/critic/critic/dense_5/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*1
_class'
%#loc:@v/critic/critic/dense_5/kernel*
valueB"      
µ
=v/critic/critic/dense_5/kernel/Adam_1/Initializer/zeros/ConstConst*1
_class'
%#loc:@v/critic/critic/dense_5/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
·
7v/critic/critic/dense_5/kernel/Adam_1/Initializer/zerosFillGv/critic/critic/dense_5/kernel/Adam_1/Initializer/zeros/shape_as_tensor=v/critic/critic/dense_5/kernel/Adam_1/Initializer/zeros/Const*
T0*1
_class'
%#loc:@v/critic/critic/dense_5/kernel*

index_type0* 
_output_shapes
:

ģ
%v/critic/critic/dense_5/kernel/Adam_1VarHandleOp*
shape:
*
dtype0*
_output_shapes
: *6
shared_name'%v/critic/critic/dense_5/kernel/Adam_1*1
_class'
%#loc:@v/critic/critic/dense_5/kernel*
	container 
Ī
Fv/critic/critic/dense_5/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp%v/critic/critic/dense_5/kernel/Adam_1*1
_class'
%#loc:@v/critic/critic/dense_5/kernel*
_output_shapes
: 
ą
,v/critic/critic/dense_5/kernel/Adam_1/AssignAssignVariableOp%v/critic/critic/dense_5/kernel/Adam_17v/critic/critic/dense_5/kernel/Adam_1/Initializer/zeros*1
_class'
%#loc:@v/critic/critic/dense_5/kernel*
dtype0
Ō
9v/critic/critic/dense_5/kernel/Adam_1/Read/ReadVariableOpReadVariableOp%v/critic/critic/dense_5/kernel/Adam_1*1
_class'
%#loc:@v/critic/critic/dense_5/kernel*
dtype0* 
_output_shapes
:

³
3v/critic/critic/dense_5/bias/Adam/Initializer/zerosConst*/
_class%
#!loc:@v/critic/critic/dense_5/bias*
valueB*    *
dtype0*
_output_shapes	
:
Ż
!v/critic/critic/dense_5/bias/AdamVarHandleOp*
shape:*
dtype0*
_output_shapes
: *2
shared_name#!v/critic/critic/dense_5/bias/Adam*/
_class%
#!loc:@v/critic/critic/dense_5/bias*
	container 
Ä
Bv/critic/critic/dense_5/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp!v/critic/critic/dense_5/bias/Adam*/
_class%
#!loc:@v/critic/critic/dense_5/bias*
_output_shapes
: 
Ņ
(v/critic/critic/dense_5/bias/Adam/AssignAssignVariableOp!v/critic/critic/dense_5/bias/Adam3v/critic/critic/dense_5/bias/Adam/Initializer/zeros*/
_class%
#!loc:@v/critic/critic/dense_5/bias*
dtype0
Å
5v/critic/critic/dense_5/bias/Adam/Read/ReadVariableOpReadVariableOp!v/critic/critic/dense_5/bias/Adam*/
_class%
#!loc:@v/critic/critic/dense_5/bias*
dtype0*
_output_shapes	
:
µ
5v/critic/critic/dense_5/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:*/
_class%
#!loc:@v/critic/critic/dense_5/bias*
valueB*    *
dtype0
į
#v/critic/critic/dense_5/bias/Adam_1VarHandleOp*
dtype0*
_output_shapes
: *4
shared_name%#v/critic/critic/dense_5/bias/Adam_1*/
_class%
#!loc:@v/critic/critic/dense_5/bias*
	container *
shape:
Č
Dv/critic/critic/dense_5/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp#v/critic/critic/dense_5/bias/Adam_1*/
_class%
#!loc:@v/critic/critic/dense_5/bias*
_output_shapes
: 
Ų
*v/critic/critic/dense_5/bias/Adam_1/AssignAssignVariableOp#v/critic/critic/dense_5/bias/Adam_15v/critic/critic/dense_5/bias/Adam_1/Initializer/zeros*/
_class%
#!loc:@v/critic/critic/dense_5/bias*
dtype0
É
7v/critic/critic/dense_5/bias/Adam_1/Read/ReadVariableOpReadVariableOp#v/critic/critic/dense_5/bias/Adam_1*
_output_shapes	
:*/
_class%
#!loc:@v/critic/critic/dense_5/bias*
dtype0
æ
5v/critic/critic/dense_6/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:	*1
_class'
%#loc:@v/critic/critic/dense_6/kernel*
valueB	*    
ē
#v/critic/critic/dense_6/kernel/AdamVarHandleOp*
	container *
shape:	*
dtype0*
_output_shapes
: *4
shared_name%#v/critic/critic/dense_6/kernel/Adam*1
_class'
%#loc:@v/critic/critic/dense_6/kernel
Ź
Dv/critic/critic/dense_6/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp#v/critic/critic/dense_6/kernel/Adam*1
_class'
%#loc:@v/critic/critic/dense_6/kernel*
_output_shapes
: 
Ś
*v/critic/critic/dense_6/kernel/Adam/AssignAssignVariableOp#v/critic/critic/dense_6/kernel/Adam5v/critic/critic/dense_6/kernel/Adam/Initializer/zeros*1
_class'
%#loc:@v/critic/critic/dense_6/kernel*
dtype0
Ļ
7v/critic/critic/dense_6/kernel/Adam/Read/ReadVariableOpReadVariableOp#v/critic/critic/dense_6/kernel/Adam*1
_class'
%#loc:@v/critic/critic/dense_6/kernel*
dtype0*
_output_shapes
:	
Į
7v/critic/critic/dense_6/kernel/Adam_1/Initializer/zerosConst*1
_class'
%#loc:@v/critic/critic/dense_6/kernel*
valueB	*    *
dtype0*
_output_shapes
:	
ė
%v/critic/critic/dense_6/kernel/Adam_1VarHandleOp*
dtype0*
_output_shapes
: *6
shared_name'%v/critic/critic/dense_6/kernel/Adam_1*1
_class'
%#loc:@v/critic/critic/dense_6/kernel*
	container *
shape:	
Ī
Fv/critic/critic/dense_6/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp%v/critic/critic/dense_6/kernel/Adam_1*1
_class'
%#loc:@v/critic/critic/dense_6/kernel*
_output_shapes
: 
ą
,v/critic/critic/dense_6/kernel/Adam_1/AssignAssignVariableOp%v/critic/critic/dense_6/kernel/Adam_17v/critic/critic/dense_6/kernel/Adam_1/Initializer/zeros*1
_class'
%#loc:@v/critic/critic/dense_6/kernel*
dtype0
Ó
9v/critic/critic/dense_6/kernel/Adam_1/Read/ReadVariableOpReadVariableOp%v/critic/critic/dense_6/kernel/Adam_1*1
_class'
%#loc:@v/critic/critic/dense_6/kernel*
dtype0*
_output_shapes
:	
±
3v/critic/critic/dense_6/bias/Adam/Initializer/zerosConst*/
_class%
#!loc:@v/critic/critic/dense_6/bias*
valueB*    *
dtype0*
_output_shapes
:
Ü
!v/critic/critic/dense_6/bias/AdamVarHandleOp*
dtype0*
_output_shapes
: *2
shared_name#!v/critic/critic/dense_6/bias/Adam*/
_class%
#!loc:@v/critic/critic/dense_6/bias*
	container *
shape:
Ä
Bv/critic/critic/dense_6/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOp!v/critic/critic/dense_6/bias/Adam*
_output_shapes
: */
_class%
#!loc:@v/critic/critic/dense_6/bias
Ņ
(v/critic/critic/dense_6/bias/Adam/AssignAssignVariableOp!v/critic/critic/dense_6/bias/Adam3v/critic/critic/dense_6/bias/Adam/Initializer/zeros*/
_class%
#!loc:@v/critic/critic/dense_6/bias*
dtype0
Ä
5v/critic/critic/dense_6/bias/Adam/Read/ReadVariableOpReadVariableOp!v/critic/critic/dense_6/bias/Adam*/
_class%
#!loc:@v/critic/critic/dense_6/bias*
dtype0*
_output_shapes
:
³
5v/critic/critic/dense_6/bias/Adam_1/Initializer/zerosConst*/
_class%
#!loc:@v/critic/critic/dense_6/bias*
valueB*    *
dtype0*
_output_shapes
:
ą
#v/critic/critic/dense_6/bias/Adam_1VarHandleOp*
dtype0*
_output_shapes
: *4
shared_name%#v/critic/critic/dense_6/bias/Adam_1*/
_class%
#!loc:@v/critic/critic/dense_6/bias*
	container *
shape:
Č
Dv/critic/critic/dense_6/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp#v/critic/critic/dense_6/bias/Adam_1*/
_class%
#!loc:@v/critic/critic/dense_6/bias*
_output_shapes
: 
Ų
*v/critic/critic/dense_6/bias/Adam_1/AssignAssignVariableOp#v/critic/critic/dense_6/bias/Adam_15v/critic/critic/dense_6/bias/Adam_1/Initializer/zeros*/
_class%
#!loc:@v/critic/critic/dense_6/bias*
dtype0
Č
7v/critic/critic/dense_6/bias/Adam_1/Read/ReadVariableOpReadVariableOp#v/critic/critic/dense_6/bias/Adam_1*/
_class%
#!loc:@v/critic/critic/dense_6/bias*
dtype0*
_output_shapes
:
Y
Adam_1/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
Q
Adam_1/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_1/beta2Const*
valueB
 *w¾?*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
valueB
 *wĢ+2*
dtype0*
_output_shapes
: 

MAdam_1/update_v/critic/critic/dense_4/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power_1*
dtype0*
_output_shapes
: 

OAdam_1/update_v/critic/critic/dense_4/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power_1*
dtype0*
_output_shapes
: 
“
>Adam_1/update_v/critic/critic/dense_4/kernel/ResourceApplyAdamResourceApplyAdamv/critic/critic/dense_4/kernel#v/critic/critic/dense_4/kernel/Adam%v/critic/critic/dense_4/kernel/Adam_1MAdam_1/update_v/critic/critic/dense_4/kernel/ResourceApplyAdam/ReadVariableOpOAdam_1/update_v/critic/critic/dense_4/kernel/ResourceApplyAdam/ReadVariableOp_1Adam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon.clip_by_global_norm_1/clip_by_global_norm_1/_0*
use_locking( *
T0*1
_class'
%#loc:@v/critic/critic/dense_4/kernel*
use_nesterov( 

KAdam_1/update_v/critic/critic/dense_4/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power_1*
dtype0*
_output_shapes
: 

MAdam_1/update_v/critic/critic/dense_4/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power_1*
dtype0*
_output_shapes
: 
¦
<Adam_1/update_v/critic/critic/dense_4/bias/ResourceApplyAdamResourceApplyAdamv/critic/critic/dense_4/bias!v/critic/critic/dense_4/bias/Adam#v/critic/critic/dense_4/bias/Adam_1KAdam_1/update_v/critic/critic/dense_4/bias/ResourceApplyAdam/ReadVariableOpMAdam_1/update_v/critic/critic/dense_4/bias/ResourceApplyAdam/ReadVariableOp_1Adam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon.clip_by_global_norm_1/clip_by_global_norm_1/_1*
use_nesterov( *
use_locking( *
T0*/
_class%
#!loc:@v/critic/critic/dense_4/bias

MAdam_1/update_v/critic/critic/dense_5/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power_1*
dtype0*
_output_shapes
: 

OAdam_1/update_v/critic/critic/dense_5/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power_1*
dtype0*
_output_shapes
: 
“
>Adam_1/update_v/critic/critic/dense_5/kernel/ResourceApplyAdamResourceApplyAdamv/critic/critic/dense_5/kernel#v/critic/critic/dense_5/kernel/Adam%v/critic/critic/dense_5/kernel/Adam_1MAdam_1/update_v/critic/critic/dense_5/kernel/ResourceApplyAdam/ReadVariableOpOAdam_1/update_v/critic/critic/dense_5/kernel/ResourceApplyAdam/ReadVariableOp_1Adam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon.clip_by_global_norm_1/clip_by_global_norm_1/_2*
use_locking( *
T0*1
_class'
%#loc:@v/critic/critic/dense_5/kernel*
use_nesterov( 

KAdam_1/update_v/critic/critic/dense_5/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power_1*
dtype0*
_output_shapes
: 

MAdam_1/update_v/critic/critic/dense_5/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power_1*
_output_shapes
: *
dtype0
¦
<Adam_1/update_v/critic/critic/dense_5/bias/ResourceApplyAdamResourceApplyAdamv/critic/critic/dense_5/bias!v/critic/critic/dense_5/bias/Adam#v/critic/critic/dense_5/bias/Adam_1KAdam_1/update_v/critic/critic/dense_5/bias/ResourceApplyAdam/ReadVariableOpMAdam_1/update_v/critic/critic/dense_5/bias/ResourceApplyAdam/ReadVariableOp_1Adam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon.clip_by_global_norm_1/clip_by_global_norm_1/_3*
use_nesterov( *
use_locking( *
T0*/
_class%
#!loc:@v/critic/critic/dense_5/bias

MAdam_1/update_v/critic/critic/dense_6/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power_1*
_output_shapes
: *
dtype0

OAdam_1/update_v/critic/critic/dense_6/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power_1*
_output_shapes
: *
dtype0
“
>Adam_1/update_v/critic/critic/dense_6/kernel/ResourceApplyAdamResourceApplyAdamv/critic/critic/dense_6/kernel#v/critic/critic/dense_6/kernel/Adam%v/critic/critic/dense_6/kernel/Adam_1MAdam_1/update_v/critic/critic/dense_6/kernel/ResourceApplyAdam/ReadVariableOpOAdam_1/update_v/critic/critic/dense_6/kernel/ResourceApplyAdam/ReadVariableOp_1Adam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon.clip_by_global_norm_1/clip_by_global_norm_1/_4*
use_locking( *
T0*1
_class'
%#loc:@v/critic/critic/dense_6/kernel*
use_nesterov( 

KAdam_1/update_v/critic/critic/dense_6/bias/ResourceApplyAdam/ReadVariableOpReadVariableOpbeta1_power_1*
dtype0*
_output_shapes
: 

MAdam_1/update_v/critic/critic/dense_6/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOpbeta2_power_1*
_output_shapes
: *
dtype0
¦
<Adam_1/update_v/critic/critic/dense_6/bias/ResourceApplyAdamResourceApplyAdamv/critic/critic/dense_6/bias!v/critic/critic/dense_6/bias/Adam#v/critic/critic/dense_6/bias/Adam_1KAdam_1/update_v/critic/critic/dense_6/bias/ResourceApplyAdam/ReadVariableOpMAdam_1/update_v/critic/critic/dense_6/bias/ResourceApplyAdam/ReadVariableOp_1Adam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon.clip_by_global_norm_1/clip_by_global_norm_1/_5*
use_locking( *
T0*/
_class%
#!loc:@v/critic/critic/dense_6/bias*
use_nesterov( 
Ū
Adam_1/ReadVariableOpReadVariableOpbeta1_power_1=^Adam_1/update_v/critic/critic/dense_4/bias/ResourceApplyAdam?^Adam_1/update_v/critic/critic/dense_4/kernel/ResourceApplyAdam=^Adam_1/update_v/critic/critic/dense_5/bias/ResourceApplyAdam?^Adam_1/update_v/critic/critic/dense_5/kernel/ResourceApplyAdam=^Adam_1/update_v/critic/critic/dense_6/bias/ResourceApplyAdam?^Adam_1/update_v/critic/critic/dense_6/kernel/ResourceApplyAdam*
dtype0*
_output_shapes
: 


Adam_1/mulMulAdam_1/ReadVariableOpAdam_1/beta1*/
_class%
#!loc:@v/critic/critic/dense_4/bias*
_output_shapes
: *
T0

Adam_1/AssignVariableOpAssignVariableOpbeta1_power_1
Adam_1/mul*/
_class%
#!loc:@v/critic/critic/dense_4/bias*
dtype0
Ø
Adam_1/ReadVariableOp_1ReadVariableOpbeta1_power_1^Adam_1/AssignVariableOp=^Adam_1/update_v/critic/critic/dense_4/bias/ResourceApplyAdam?^Adam_1/update_v/critic/critic/dense_4/kernel/ResourceApplyAdam=^Adam_1/update_v/critic/critic/dense_5/bias/ResourceApplyAdam?^Adam_1/update_v/critic/critic/dense_5/kernel/ResourceApplyAdam=^Adam_1/update_v/critic/critic/dense_6/bias/ResourceApplyAdam?^Adam_1/update_v/critic/critic/dense_6/kernel/ResourceApplyAdam*/
_class%
#!loc:@v/critic/critic/dense_4/bias*
dtype0*
_output_shapes
: 
Ż
Adam_1/ReadVariableOp_2ReadVariableOpbeta2_power_1=^Adam_1/update_v/critic/critic/dense_4/bias/ResourceApplyAdam?^Adam_1/update_v/critic/critic/dense_4/kernel/ResourceApplyAdam=^Adam_1/update_v/critic/critic/dense_5/bias/ResourceApplyAdam?^Adam_1/update_v/critic/critic/dense_5/kernel/ResourceApplyAdam=^Adam_1/update_v/critic/critic/dense_6/bias/ResourceApplyAdam?^Adam_1/update_v/critic/critic/dense_6/kernel/ResourceApplyAdam*
_output_shapes
: *
dtype0

Adam_1/mul_1MulAdam_1/ReadVariableOp_2Adam_1/beta2*
_output_shapes
: *
T0*/
_class%
#!loc:@v/critic/critic/dense_4/bias

Adam_1/AssignVariableOp_1AssignVariableOpbeta2_power_1Adam_1/mul_1*
dtype0*/
_class%
#!loc:@v/critic/critic/dense_4/bias
Ŗ
Adam_1/ReadVariableOp_3ReadVariableOpbeta2_power_1^Adam_1/AssignVariableOp_1=^Adam_1/update_v/critic/critic/dense_4/bias/ResourceApplyAdam?^Adam_1/update_v/critic/critic/dense_4/kernel/ResourceApplyAdam=^Adam_1/update_v/critic/critic/dense_5/bias/ResourceApplyAdam?^Adam_1/update_v/critic/critic/dense_5/kernel/ResourceApplyAdam=^Adam_1/update_v/critic/critic/dense_6/bias/ResourceApplyAdam?^Adam_1/update_v/critic/critic/dense_6/kernel/ResourceApplyAdam*/
_class%
#!loc:@v/critic/critic/dense_4/bias*
dtype0*
_output_shapes
: 
Ä
Adam_1NoOp^Adam_1/AssignVariableOp^Adam_1/AssignVariableOp_1=^Adam_1/update_v/critic/critic/dense_4/bias/ResourceApplyAdam?^Adam_1/update_v/critic/critic/dense_4/kernel/ResourceApplyAdam=^Adam_1/update_v/critic/critic/dense_5/bias/ResourceApplyAdam?^Adam_1/update_v/critic/critic/dense_5/kernel/ResourceApplyAdam=^Adam_1/update_v/critic/critic/dense_6/bias/ResourceApplyAdam?^Adam_1/update_v/critic/critic/dense_6/kernel/ResourceApplyAdam
Ī
initNoOp^beta1_power/Assign^beta1_power_1/Assign^beta2_power/Assign^beta2_power_1/Assign&^pi/actor/actor/dense/bias/Adam/Assign(^pi/actor/actor/dense/bias/Adam_1/Assign!^pi/actor/actor/dense/bias/Assign(^pi/actor/actor/dense/kernel/Adam/Assign*^pi/actor/actor/dense/kernel/Adam_1/Assign#^pi/actor/actor/dense/kernel/Assign(^pi/actor/actor/dense_1/bias/Adam/Assign*^pi/actor/actor/dense_1/bias/Adam_1/Assign#^pi/actor/actor/dense_1/bias/Assign*^pi/actor/actor/dense_1/kernel/Adam/Assign,^pi/actor/actor/dense_1/kernel/Adam_1/Assign%^pi/actor/actor/dense_1/kernel/Assign(^pi/actor/actor/dense_2/bias/Adam/Assign*^pi/actor/actor/dense_2/bias/Adam_1/Assign#^pi/actor/actor/dense_2/bias/Assign*^pi/actor/actor/dense_2/kernel/Adam/Assign,^pi/actor/actor/dense_2/kernel/Adam_1/Assign%^pi/actor/actor/dense_2/kernel/Assign(^pi/actor/actor/dense_3/bias/Adam/Assign*^pi/actor/actor/dense_3/bias/Adam_1/Assign#^pi/actor/actor/dense_3/bias/Assign*^pi/actor/actor/dense_3/kernel/Adam/Assign,^pi/actor/actor/dense_3/kernel/Adam_1/Assign%^pi/actor/actor/dense_3/kernel/Assign)^v/critic/critic/dense_4/bias/Adam/Assign+^v/critic/critic/dense_4/bias/Adam_1/Assign$^v/critic/critic/dense_4/bias/Assign+^v/critic/critic/dense_4/kernel/Adam/Assign-^v/critic/critic/dense_4/kernel/Adam_1/Assign&^v/critic/critic/dense_4/kernel/Assign)^v/critic/critic/dense_5/bias/Adam/Assign+^v/critic/critic/dense_5/bias/Adam_1/Assign$^v/critic/critic/dense_5/bias/Assign+^v/critic/critic/dense_5/kernel/Adam/Assign-^v/critic/critic/dense_5/kernel/Adam_1/Assign&^v/critic/critic/dense_5/kernel/Assign)^v/critic/critic/dense_6/bias/Adam/Assign+^v/critic/critic/dense_6/bias/Adam_1/Assign$^v/critic/critic/dense_6/bias/Assign+^v/critic/critic/dense_6/kernel/Adam/Assign-^v/critic/critic/dense_6/kernel/Adam_1/Assign&^v/critic/critic/dense_6/kernel/Assign
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_d6cfe0bbe8df49d497b0ee009e5c1571/part
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
\
save/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
“
save/SaveV2/tensor_namesConst*ē
valueŻBŚ.Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/actor/actor/dense/biasBpi/actor/actor/dense/bias/AdamB pi/actor/actor/dense/bias/Adam_1Bpi/actor/actor/dense/kernelB pi/actor/actor/dense/kernel/AdamB"pi/actor/actor/dense/kernel/Adam_1Bpi/actor/actor/dense_1/biasB pi/actor/actor/dense_1/bias/AdamB"pi/actor/actor/dense_1/bias/Adam_1Bpi/actor/actor/dense_1/kernelB"pi/actor/actor/dense_1/kernel/AdamB$pi/actor/actor/dense_1/kernel/Adam_1Bpi/actor/actor/dense_2/biasB pi/actor/actor/dense_2/bias/AdamB"pi/actor/actor/dense_2/bias/Adam_1Bpi/actor/actor/dense_2/kernelB"pi/actor/actor/dense_2/kernel/AdamB$pi/actor/actor/dense_2/kernel/Adam_1Bpi/actor/actor/dense_3/biasB pi/actor/actor/dense_3/bias/AdamB"pi/actor/actor/dense_3/bias/Adam_1Bpi/actor/actor/dense_3/kernelB"pi/actor/actor/dense_3/kernel/AdamB$pi/actor/actor/dense_3/kernel/Adam_1Bv/critic/critic/dense_4/biasB!v/critic/critic/dense_4/bias/AdamB#v/critic/critic/dense_4/bias/Adam_1Bv/critic/critic/dense_4/kernelB#v/critic/critic/dense_4/kernel/AdamB%v/critic/critic/dense_4/kernel/Adam_1Bv/critic/critic/dense_5/biasB!v/critic/critic/dense_5/bias/AdamB#v/critic/critic/dense_5/bias/Adam_1Bv/critic/critic/dense_5/kernelB#v/critic/critic/dense_5/kernel/AdamB%v/critic/critic/dense_5/kernel/Adam_1Bv/critic/critic/dense_6/biasB!v/critic/critic/dense_6/bias/AdamB#v/critic/critic/dense_6/bias/Adam_1Bv/critic/critic/dense_6/kernelB#v/critic/critic/dense_6/kernel/AdamB%v/critic/critic/dense_6/kernel/Adam_1*
dtype0*
_output_shapes
:.
æ
save/SaveV2/shape_and_slicesConst*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:.

save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_power/Read/ReadVariableOp!beta1_power_1/Read/ReadVariableOpbeta2_power/Read/ReadVariableOp!beta2_power_1/Read/ReadVariableOp-pi/actor/actor/dense/bias/Read/ReadVariableOp2pi/actor/actor/dense/bias/Adam/Read/ReadVariableOp4pi/actor/actor/dense/bias/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense/kernel/Read/ReadVariableOp4pi/actor/actor/dense/kernel/Adam/Read/ReadVariableOp6pi/actor/actor/dense/kernel/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense_1/bias/Read/ReadVariableOp4pi/actor/actor/dense_1/bias/Adam/Read/ReadVariableOp6pi/actor/actor/dense_1/bias/Adam_1/Read/ReadVariableOp1pi/actor/actor/dense_1/kernel/Read/ReadVariableOp6pi/actor/actor/dense_1/kernel/Adam/Read/ReadVariableOp8pi/actor/actor/dense_1/kernel/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense_2/bias/Read/ReadVariableOp4pi/actor/actor/dense_2/bias/Adam/Read/ReadVariableOp6pi/actor/actor/dense_2/bias/Adam_1/Read/ReadVariableOp1pi/actor/actor/dense_2/kernel/Read/ReadVariableOp6pi/actor/actor/dense_2/kernel/Adam/Read/ReadVariableOp8pi/actor/actor/dense_2/kernel/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense_3/bias/Read/ReadVariableOp4pi/actor/actor/dense_3/bias/Adam/Read/ReadVariableOp6pi/actor/actor/dense_3/bias/Adam_1/Read/ReadVariableOp1pi/actor/actor/dense_3/kernel/Read/ReadVariableOp6pi/actor/actor/dense_3/kernel/Adam/Read/ReadVariableOp8pi/actor/actor/dense_3/kernel/Adam_1/Read/ReadVariableOp0v/critic/critic/dense_4/bias/Read/ReadVariableOp5v/critic/critic/dense_4/bias/Adam/Read/ReadVariableOp7v/critic/critic/dense_4/bias/Adam_1/Read/ReadVariableOp2v/critic/critic/dense_4/kernel/Read/ReadVariableOp7v/critic/critic/dense_4/kernel/Adam/Read/ReadVariableOp9v/critic/critic/dense_4/kernel/Adam_1/Read/ReadVariableOp0v/critic/critic/dense_5/bias/Read/ReadVariableOp5v/critic/critic/dense_5/bias/Adam/Read/ReadVariableOp7v/critic/critic/dense_5/bias/Adam_1/Read/ReadVariableOp2v/critic/critic/dense_5/kernel/Read/ReadVariableOp7v/critic/critic/dense_5/kernel/Adam/Read/ReadVariableOp9v/critic/critic/dense_5/kernel/Adam_1/Read/ReadVariableOp0v/critic/critic/dense_6/bias/Read/ReadVariableOp5v/critic/critic/dense_6/bias/Adam/Read/ReadVariableOp7v/critic/critic/dense_6/bias/Adam_1/Read/ReadVariableOp2v/critic/critic/dense_6/kernel/Read/ReadVariableOp7v/critic/critic/dense_6/kernel/Adam/Read/ReadVariableOp9v/critic/critic/dense_6/kernel/Adam_1/Read/ReadVariableOp*<
dtypes2
02.

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
N*
_output_shapes
:*
T0*

axis 
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
·
save/RestoreV2/tensor_namesConst*
_output_shapes
:.*ē
valueŻBŚ.Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/actor/actor/dense/biasBpi/actor/actor/dense/bias/AdamB pi/actor/actor/dense/bias/Adam_1Bpi/actor/actor/dense/kernelB pi/actor/actor/dense/kernel/AdamB"pi/actor/actor/dense/kernel/Adam_1Bpi/actor/actor/dense_1/biasB pi/actor/actor/dense_1/bias/AdamB"pi/actor/actor/dense_1/bias/Adam_1Bpi/actor/actor/dense_1/kernelB"pi/actor/actor/dense_1/kernel/AdamB$pi/actor/actor/dense_1/kernel/Adam_1Bpi/actor/actor/dense_2/biasB pi/actor/actor/dense_2/bias/AdamB"pi/actor/actor/dense_2/bias/Adam_1Bpi/actor/actor/dense_2/kernelB"pi/actor/actor/dense_2/kernel/AdamB$pi/actor/actor/dense_2/kernel/Adam_1Bpi/actor/actor/dense_3/biasB pi/actor/actor/dense_3/bias/AdamB"pi/actor/actor/dense_3/bias/Adam_1Bpi/actor/actor/dense_3/kernelB"pi/actor/actor/dense_3/kernel/AdamB$pi/actor/actor/dense_3/kernel/Adam_1Bv/critic/critic/dense_4/biasB!v/critic/critic/dense_4/bias/AdamB#v/critic/critic/dense_4/bias/Adam_1Bv/critic/critic/dense_4/kernelB#v/critic/critic/dense_4/kernel/AdamB%v/critic/critic/dense_4/kernel/Adam_1Bv/critic/critic/dense_5/biasB!v/critic/critic/dense_5/bias/AdamB#v/critic/critic/dense_5/bias/Adam_1Bv/critic/critic/dense_5/kernelB#v/critic/critic/dense_5/kernel/AdamB%v/critic/critic/dense_5/kernel/Adam_1Bv/critic/critic/dense_6/biasB!v/critic/critic/dense_6/bias/AdamB#v/critic/critic/dense_6/bias/Adam_1Bv/critic/critic/dense_6/kernelB#v/critic/critic/dense_6/kernel/AdamB%v/critic/critic/dense_6/kernel/Adam_1*
dtype0
Ā
save/RestoreV2/shape_and_slicesConst*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:.
ō
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*Ī
_output_shapes»
ø::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.
N
save/Identity_1Identitysave/RestoreV2*
T0*
_output_shapes
:
T
save/AssignVariableOpAssignVariableOpbeta1_powersave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
X
save/AssignVariableOp_1AssignVariableOpbeta1_power_1save/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
_output_shapes
:*
T0
V
save/AssignVariableOp_2AssignVariableOpbeta2_powersave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
T0*
_output_shapes
:
X
save/AssignVariableOp_3AssignVariableOpbeta2_power_1save/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
d
save/AssignVariableOp_4AssignVariableOppi/actor/actor/dense/biassave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
T0*
_output_shapes
:
i
save/AssignVariableOp_5AssignVariableOppi/actor/actor/dense/bias/Adamsave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
T0*
_output_shapes
:
k
save/AssignVariableOp_6AssignVariableOp pi/actor/actor/dense/bias/Adam_1save/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:7*
T0*
_output_shapes
:
f
save/AssignVariableOp_7AssignVariableOppi/actor/actor/dense/kernelsave/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:8*
T0*
_output_shapes
:
k
save/AssignVariableOp_8AssignVariableOp pi/actor/actor/dense/kernel/Adamsave/Identity_9*
dtype0
Q
save/Identity_10Identitysave/RestoreV2:9*
_output_shapes
:*
T0
n
save/AssignVariableOp_9AssignVariableOp"pi/actor/actor/dense/kernel/Adam_1save/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:10*
T0*
_output_shapes
:
h
save/AssignVariableOp_10AssignVariableOppi/actor/actor/dense_1/biassave/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:11*
T0*
_output_shapes
:
m
save/AssignVariableOp_11AssignVariableOp pi/actor/actor/dense_1/bias/Adamsave/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:12*
T0*
_output_shapes
:
o
save/AssignVariableOp_12AssignVariableOp"pi/actor/actor/dense_1/bias/Adam_1save/Identity_13*
dtype0
R
save/Identity_14Identitysave/RestoreV2:13*
T0*
_output_shapes
:
j
save/AssignVariableOp_13AssignVariableOppi/actor/actor/dense_1/kernelsave/Identity_14*
dtype0
R
save/Identity_15Identitysave/RestoreV2:14*
T0*
_output_shapes
:
o
save/AssignVariableOp_14AssignVariableOp"pi/actor/actor/dense_1/kernel/Adamsave/Identity_15*
dtype0
R
save/Identity_16Identitysave/RestoreV2:15*
T0*
_output_shapes
:
q
save/AssignVariableOp_15AssignVariableOp$pi/actor/actor/dense_1/kernel/Adam_1save/Identity_16*
dtype0
R
save/Identity_17Identitysave/RestoreV2:16*
T0*
_output_shapes
:
h
save/AssignVariableOp_16AssignVariableOppi/actor/actor/dense_2/biassave/Identity_17*
dtype0
R
save/Identity_18Identitysave/RestoreV2:17*
T0*
_output_shapes
:
m
save/AssignVariableOp_17AssignVariableOp pi/actor/actor/dense_2/bias/Adamsave/Identity_18*
dtype0
R
save/Identity_19Identitysave/RestoreV2:18*
T0*
_output_shapes
:
o
save/AssignVariableOp_18AssignVariableOp"pi/actor/actor/dense_2/bias/Adam_1save/Identity_19*
dtype0
R
save/Identity_20Identitysave/RestoreV2:19*
_output_shapes
:*
T0
j
save/AssignVariableOp_19AssignVariableOppi/actor/actor/dense_2/kernelsave/Identity_20*
dtype0
R
save/Identity_21Identitysave/RestoreV2:20*
_output_shapes
:*
T0
o
save/AssignVariableOp_20AssignVariableOp"pi/actor/actor/dense_2/kernel/Adamsave/Identity_21*
dtype0
R
save/Identity_22Identitysave/RestoreV2:21*
_output_shapes
:*
T0
q
save/AssignVariableOp_21AssignVariableOp$pi/actor/actor/dense_2/kernel/Adam_1save/Identity_22*
dtype0
R
save/Identity_23Identitysave/RestoreV2:22*
T0*
_output_shapes
:
h
save/AssignVariableOp_22AssignVariableOppi/actor/actor/dense_3/biassave/Identity_23*
dtype0
R
save/Identity_24Identitysave/RestoreV2:23*
_output_shapes
:*
T0
m
save/AssignVariableOp_23AssignVariableOp pi/actor/actor/dense_3/bias/Adamsave/Identity_24*
dtype0
R
save/Identity_25Identitysave/RestoreV2:24*
_output_shapes
:*
T0
o
save/AssignVariableOp_24AssignVariableOp"pi/actor/actor/dense_3/bias/Adam_1save/Identity_25*
dtype0
R
save/Identity_26Identitysave/RestoreV2:25*
T0*
_output_shapes
:
j
save/AssignVariableOp_25AssignVariableOppi/actor/actor/dense_3/kernelsave/Identity_26*
dtype0
R
save/Identity_27Identitysave/RestoreV2:26*
T0*
_output_shapes
:
o
save/AssignVariableOp_26AssignVariableOp"pi/actor/actor/dense_3/kernel/Adamsave/Identity_27*
dtype0
R
save/Identity_28Identitysave/RestoreV2:27*
_output_shapes
:*
T0
q
save/AssignVariableOp_27AssignVariableOp$pi/actor/actor/dense_3/kernel/Adam_1save/Identity_28*
dtype0
R
save/Identity_29Identitysave/RestoreV2:28*
T0*
_output_shapes
:
i
save/AssignVariableOp_28AssignVariableOpv/critic/critic/dense_4/biassave/Identity_29*
dtype0
R
save/Identity_30Identitysave/RestoreV2:29*
T0*
_output_shapes
:
n
save/AssignVariableOp_29AssignVariableOp!v/critic/critic/dense_4/bias/Adamsave/Identity_30*
dtype0
R
save/Identity_31Identitysave/RestoreV2:30*
T0*
_output_shapes
:
p
save/AssignVariableOp_30AssignVariableOp#v/critic/critic/dense_4/bias/Adam_1save/Identity_31*
dtype0
R
save/Identity_32Identitysave/RestoreV2:31*
T0*
_output_shapes
:
k
save/AssignVariableOp_31AssignVariableOpv/critic/critic/dense_4/kernelsave/Identity_32*
dtype0
R
save/Identity_33Identitysave/RestoreV2:32*
T0*
_output_shapes
:
p
save/AssignVariableOp_32AssignVariableOp#v/critic/critic/dense_4/kernel/Adamsave/Identity_33*
dtype0
R
save/Identity_34Identitysave/RestoreV2:33*
_output_shapes
:*
T0
r
save/AssignVariableOp_33AssignVariableOp%v/critic/critic/dense_4/kernel/Adam_1save/Identity_34*
dtype0
R
save/Identity_35Identitysave/RestoreV2:34*
T0*
_output_shapes
:
i
save/AssignVariableOp_34AssignVariableOpv/critic/critic/dense_5/biassave/Identity_35*
dtype0
R
save/Identity_36Identitysave/RestoreV2:35*
T0*
_output_shapes
:
n
save/AssignVariableOp_35AssignVariableOp!v/critic/critic/dense_5/bias/Adamsave/Identity_36*
dtype0
R
save/Identity_37Identitysave/RestoreV2:36*
_output_shapes
:*
T0
p
save/AssignVariableOp_36AssignVariableOp#v/critic/critic/dense_5/bias/Adam_1save/Identity_37*
dtype0
R
save/Identity_38Identitysave/RestoreV2:37*
_output_shapes
:*
T0
k
save/AssignVariableOp_37AssignVariableOpv/critic/critic/dense_5/kernelsave/Identity_38*
dtype0
R
save/Identity_39Identitysave/RestoreV2:38*
T0*
_output_shapes
:
p
save/AssignVariableOp_38AssignVariableOp#v/critic/critic/dense_5/kernel/Adamsave/Identity_39*
dtype0
R
save/Identity_40Identitysave/RestoreV2:39*
_output_shapes
:*
T0
r
save/AssignVariableOp_39AssignVariableOp%v/critic/critic/dense_5/kernel/Adam_1save/Identity_40*
dtype0
R
save/Identity_41Identitysave/RestoreV2:40*
_output_shapes
:*
T0
i
save/AssignVariableOp_40AssignVariableOpv/critic/critic/dense_6/biassave/Identity_41*
dtype0
R
save/Identity_42Identitysave/RestoreV2:41*
_output_shapes
:*
T0
n
save/AssignVariableOp_41AssignVariableOp!v/critic/critic/dense_6/bias/Adamsave/Identity_42*
dtype0
R
save/Identity_43Identitysave/RestoreV2:42*
T0*
_output_shapes
:
p
save/AssignVariableOp_42AssignVariableOp#v/critic/critic/dense_6/bias/Adam_1save/Identity_43*
dtype0
R
save/Identity_44Identitysave/RestoreV2:43*
_output_shapes
:*
T0
k
save/AssignVariableOp_43AssignVariableOpv/critic/critic/dense_6/kernelsave/Identity_44*
dtype0
R
save/Identity_45Identitysave/RestoreV2:44*
T0*
_output_shapes
:
p
save/AssignVariableOp_44AssignVariableOp#v/critic/critic/dense_6/kernel/Adamsave/Identity_45*
dtype0
R
save/Identity_46Identitysave/RestoreV2:45*
_output_shapes
:*
T0
r
save/AssignVariableOp_45AssignVariableOp%v/critic/critic/dense_6/kernel/Adam_1save/Identity_46*
dtype0
č	
save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_23^save/AssignVariableOp_24^save/AssignVariableOp_25^save/AssignVariableOp_26^save/AssignVariableOp_27^save/AssignVariableOp_28^save/AssignVariableOp_29^save/AssignVariableOp_3^save/AssignVariableOp_30^save/AssignVariableOp_31^save/AssignVariableOp_32^save/AssignVariableOp_33^save/AssignVariableOp_34^save/AssignVariableOp_35^save/AssignVariableOp_36^save/AssignVariableOp_37^save/AssignVariableOp_38^save/AssignVariableOp_39^save/AssignVariableOp_4^save/AssignVariableOp_40^save/AssignVariableOp_41^save/AssignVariableOp_42^save/AssignVariableOp_43^save/AssignVariableOp_44^save/AssignVariableOp_45^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
-
save/restore_allNoOp^save/restore_shard
[
save_1/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
_output_shapes
: *
shape: *
dtype0

save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_758bb42d508a41db827de81c5b2b8d6e/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
¶
save_1/SaveV2/tensor_namesConst*
_output_shapes
:.*ē
valueŻBŚ.Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/actor/actor/dense/biasBpi/actor/actor/dense/bias/AdamB pi/actor/actor/dense/bias/Adam_1Bpi/actor/actor/dense/kernelB pi/actor/actor/dense/kernel/AdamB"pi/actor/actor/dense/kernel/Adam_1Bpi/actor/actor/dense_1/biasB pi/actor/actor/dense_1/bias/AdamB"pi/actor/actor/dense_1/bias/Adam_1Bpi/actor/actor/dense_1/kernelB"pi/actor/actor/dense_1/kernel/AdamB$pi/actor/actor/dense_1/kernel/Adam_1Bpi/actor/actor/dense_2/biasB pi/actor/actor/dense_2/bias/AdamB"pi/actor/actor/dense_2/bias/Adam_1Bpi/actor/actor/dense_2/kernelB"pi/actor/actor/dense_2/kernel/AdamB$pi/actor/actor/dense_2/kernel/Adam_1Bpi/actor/actor/dense_3/biasB pi/actor/actor/dense_3/bias/AdamB"pi/actor/actor/dense_3/bias/Adam_1Bpi/actor/actor/dense_3/kernelB"pi/actor/actor/dense_3/kernel/AdamB$pi/actor/actor/dense_3/kernel/Adam_1Bv/critic/critic/dense_4/biasB!v/critic/critic/dense_4/bias/AdamB#v/critic/critic/dense_4/bias/Adam_1Bv/critic/critic/dense_4/kernelB#v/critic/critic/dense_4/kernel/AdamB%v/critic/critic/dense_4/kernel/Adam_1Bv/critic/critic/dense_5/biasB!v/critic/critic/dense_5/bias/AdamB#v/critic/critic/dense_5/bias/Adam_1Bv/critic/critic/dense_5/kernelB#v/critic/critic/dense_5/kernel/AdamB%v/critic/critic/dense_5/kernel/Adam_1Bv/critic/critic/dense_6/biasB!v/critic/critic/dense_6/bias/AdamB#v/critic/critic/dense_6/bias/Adam_1Bv/critic/critic/dense_6/kernelB#v/critic/critic/dense_6/kernel/AdamB%v/critic/critic/dense_6/kernel/Adam_1*
dtype0
Į
save_1/SaveV2/shape_and_slicesConst*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:.

save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_power/Read/ReadVariableOp!beta1_power_1/Read/ReadVariableOpbeta2_power/Read/ReadVariableOp!beta2_power_1/Read/ReadVariableOp-pi/actor/actor/dense/bias/Read/ReadVariableOp2pi/actor/actor/dense/bias/Adam/Read/ReadVariableOp4pi/actor/actor/dense/bias/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense/kernel/Read/ReadVariableOp4pi/actor/actor/dense/kernel/Adam/Read/ReadVariableOp6pi/actor/actor/dense/kernel/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense_1/bias/Read/ReadVariableOp4pi/actor/actor/dense_1/bias/Adam/Read/ReadVariableOp6pi/actor/actor/dense_1/bias/Adam_1/Read/ReadVariableOp1pi/actor/actor/dense_1/kernel/Read/ReadVariableOp6pi/actor/actor/dense_1/kernel/Adam/Read/ReadVariableOp8pi/actor/actor/dense_1/kernel/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense_2/bias/Read/ReadVariableOp4pi/actor/actor/dense_2/bias/Adam/Read/ReadVariableOp6pi/actor/actor/dense_2/bias/Adam_1/Read/ReadVariableOp1pi/actor/actor/dense_2/kernel/Read/ReadVariableOp6pi/actor/actor/dense_2/kernel/Adam/Read/ReadVariableOp8pi/actor/actor/dense_2/kernel/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense_3/bias/Read/ReadVariableOp4pi/actor/actor/dense_3/bias/Adam/Read/ReadVariableOp6pi/actor/actor/dense_3/bias/Adam_1/Read/ReadVariableOp1pi/actor/actor/dense_3/kernel/Read/ReadVariableOp6pi/actor/actor/dense_3/kernel/Adam/Read/ReadVariableOp8pi/actor/actor/dense_3/kernel/Adam_1/Read/ReadVariableOp0v/critic/critic/dense_4/bias/Read/ReadVariableOp5v/critic/critic/dense_4/bias/Adam/Read/ReadVariableOp7v/critic/critic/dense_4/bias/Adam_1/Read/ReadVariableOp2v/critic/critic/dense_4/kernel/Read/ReadVariableOp7v/critic/critic/dense_4/kernel/Adam/Read/ReadVariableOp9v/critic/critic/dense_4/kernel/Adam_1/Read/ReadVariableOp0v/critic/critic/dense_5/bias/Read/ReadVariableOp5v/critic/critic/dense_5/bias/Adam/Read/ReadVariableOp7v/critic/critic/dense_5/bias/Adam_1/Read/ReadVariableOp2v/critic/critic/dense_5/kernel/Read/ReadVariableOp7v/critic/critic/dense_5/kernel/Adam/Read/ReadVariableOp9v/critic/critic/dense_5/kernel/Adam_1/Read/ReadVariableOp0v/critic/critic/dense_6/bias/Read/ReadVariableOp5v/critic/critic/dense_6/bias/Adam/Read/ReadVariableOp7v/critic/critic/dense_6/bias/Adam_1/Read/ReadVariableOp2v/critic/critic/dense_6/kernel/Read/ReadVariableOp7v/critic/critic/dense_6/kernel/Adam/Read/ReadVariableOp9v/critic/critic/dense_6/kernel/Adam_1/Read/ReadVariableOp*<
dtypes2
02.

save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_1/ShardedFilename
£
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*

axis *
N*
_output_shapes
:*
T0

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(

save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
¹
save_1/RestoreV2/tensor_namesConst*ē
valueŻBŚ.Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/actor/actor/dense/biasBpi/actor/actor/dense/bias/AdamB pi/actor/actor/dense/bias/Adam_1Bpi/actor/actor/dense/kernelB pi/actor/actor/dense/kernel/AdamB"pi/actor/actor/dense/kernel/Adam_1Bpi/actor/actor/dense_1/biasB pi/actor/actor/dense_1/bias/AdamB"pi/actor/actor/dense_1/bias/Adam_1Bpi/actor/actor/dense_1/kernelB"pi/actor/actor/dense_1/kernel/AdamB$pi/actor/actor/dense_1/kernel/Adam_1Bpi/actor/actor/dense_2/biasB pi/actor/actor/dense_2/bias/AdamB"pi/actor/actor/dense_2/bias/Adam_1Bpi/actor/actor/dense_2/kernelB"pi/actor/actor/dense_2/kernel/AdamB$pi/actor/actor/dense_2/kernel/Adam_1Bpi/actor/actor/dense_3/biasB pi/actor/actor/dense_3/bias/AdamB"pi/actor/actor/dense_3/bias/Adam_1Bpi/actor/actor/dense_3/kernelB"pi/actor/actor/dense_3/kernel/AdamB$pi/actor/actor/dense_3/kernel/Adam_1Bv/critic/critic/dense_4/biasB!v/critic/critic/dense_4/bias/AdamB#v/critic/critic/dense_4/bias/Adam_1Bv/critic/critic/dense_4/kernelB#v/critic/critic/dense_4/kernel/AdamB%v/critic/critic/dense_4/kernel/Adam_1Bv/critic/critic/dense_5/biasB!v/critic/critic/dense_5/bias/AdamB#v/critic/critic/dense_5/bias/Adam_1Bv/critic/critic/dense_5/kernelB#v/critic/critic/dense_5/kernel/AdamB%v/critic/critic/dense_5/kernel/Adam_1Bv/critic/critic/dense_6/biasB!v/critic/critic/dense_6/bias/AdamB#v/critic/critic/dense_6/bias/Adam_1Bv/critic/critic/dense_6/kernelB#v/critic/critic/dense_6/kernel/AdamB%v/critic/critic/dense_6/kernel/Adam_1*
dtype0*
_output_shapes
:.
Ä
!save_1/RestoreV2/shape_and_slicesConst*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:.
ü
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*<
dtypes2
02.*Ī
_output_shapes»
ø::::::::::::::::::::::::::::::::::::::::::::::
R
save_1/Identity_1Identitysave_1/RestoreV2*
T0*
_output_shapes
:
X
save_1/AssignVariableOpAssignVariableOpbeta1_powersave_1/Identity_1*
dtype0
T
save_1/Identity_2Identitysave_1/RestoreV2:1*
T0*
_output_shapes
:
\
save_1/AssignVariableOp_1AssignVariableOpbeta1_power_1save_1/Identity_2*
dtype0
T
save_1/Identity_3Identitysave_1/RestoreV2:2*
T0*
_output_shapes
:
Z
save_1/AssignVariableOp_2AssignVariableOpbeta2_powersave_1/Identity_3*
dtype0
T
save_1/Identity_4Identitysave_1/RestoreV2:3*
_output_shapes
:*
T0
\
save_1/AssignVariableOp_3AssignVariableOpbeta2_power_1save_1/Identity_4*
dtype0
T
save_1/Identity_5Identitysave_1/RestoreV2:4*
T0*
_output_shapes
:
h
save_1/AssignVariableOp_4AssignVariableOppi/actor/actor/dense/biassave_1/Identity_5*
dtype0
T
save_1/Identity_6Identitysave_1/RestoreV2:5*
T0*
_output_shapes
:
m
save_1/AssignVariableOp_5AssignVariableOppi/actor/actor/dense/bias/Adamsave_1/Identity_6*
dtype0
T
save_1/Identity_7Identitysave_1/RestoreV2:6*
T0*
_output_shapes
:
o
save_1/AssignVariableOp_6AssignVariableOp pi/actor/actor/dense/bias/Adam_1save_1/Identity_7*
dtype0
T
save_1/Identity_8Identitysave_1/RestoreV2:7*
T0*
_output_shapes
:
j
save_1/AssignVariableOp_7AssignVariableOppi/actor/actor/dense/kernelsave_1/Identity_8*
dtype0
T
save_1/Identity_9Identitysave_1/RestoreV2:8*
_output_shapes
:*
T0
o
save_1/AssignVariableOp_8AssignVariableOp pi/actor/actor/dense/kernel/Adamsave_1/Identity_9*
dtype0
U
save_1/Identity_10Identitysave_1/RestoreV2:9*
_output_shapes
:*
T0
r
save_1/AssignVariableOp_9AssignVariableOp"pi/actor/actor/dense/kernel/Adam_1save_1/Identity_10*
dtype0
V
save_1/Identity_11Identitysave_1/RestoreV2:10*
T0*
_output_shapes
:
l
save_1/AssignVariableOp_10AssignVariableOppi/actor/actor/dense_1/biassave_1/Identity_11*
dtype0
V
save_1/Identity_12Identitysave_1/RestoreV2:11*
T0*
_output_shapes
:
q
save_1/AssignVariableOp_11AssignVariableOp pi/actor/actor/dense_1/bias/Adamsave_1/Identity_12*
dtype0
V
save_1/Identity_13Identitysave_1/RestoreV2:12*
T0*
_output_shapes
:
s
save_1/AssignVariableOp_12AssignVariableOp"pi/actor/actor/dense_1/bias/Adam_1save_1/Identity_13*
dtype0
V
save_1/Identity_14Identitysave_1/RestoreV2:13*
T0*
_output_shapes
:
n
save_1/AssignVariableOp_13AssignVariableOppi/actor/actor/dense_1/kernelsave_1/Identity_14*
dtype0
V
save_1/Identity_15Identitysave_1/RestoreV2:14*
T0*
_output_shapes
:
s
save_1/AssignVariableOp_14AssignVariableOp"pi/actor/actor/dense_1/kernel/Adamsave_1/Identity_15*
dtype0
V
save_1/Identity_16Identitysave_1/RestoreV2:15*
T0*
_output_shapes
:
u
save_1/AssignVariableOp_15AssignVariableOp$pi/actor/actor/dense_1/kernel/Adam_1save_1/Identity_16*
dtype0
V
save_1/Identity_17Identitysave_1/RestoreV2:16*
T0*
_output_shapes
:
l
save_1/AssignVariableOp_16AssignVariableOppi/actor/actor/dense_2/biassave_1/Identity_17*
dtype0
V
save_1/Identity_18Identitysave_1/RestoreV2:17*
_output_shapes
:*
T0
q
save_1/AssignVariableOp_17AssignVariableOp pi/actor/actor/dense_2/bias/Adamsave_1/Identity_18*
dtype0
V
save_1/Identity_19Identitysave_1/RestoreV2:18*
_output_shapes
:*
T0
s
save_1/AssignVariableOp_18AssignVariableOp"pi/actor/actor/dense_2/bias/Adam_1save_1/Identity_19*
dtype0
V
save_1/Identity_20Identitysave_1/RestoreV2:19*
T0*
_output_shapes
:
n
save_1/AssignVariableOp_19AssignVariableOppi/actor/actor/dense_2/kernelsave_1/Identity_20*
dtype0
V
save_1/Identity_21Identitysave_1/RestoreV2:20*
T0*
_output_shapes
:
s
save_1/AssignVariableOp_20AssignVariableOp"pi/actor/actor/dense_2/kernel/Adamsave_1/Identity_21*
dtype0
V
save_1/Identity_22Identitysave_1/RestoreV2:21*
T0*
_output_shapes
:
u
save_1/AssignVariableOp_21AssignVariableOp$pi/actor/actor/dense_2/kernel/Adam_1save_1/Identity_22*
dtype0
V
save_1/Identity_23Identitysave_1/RestoreV2:22*
_output_shapes
:*
T0
l
save_1/AssignVariableOp_22AssignVariableOppi/actor/actor/dense_3/biassave_1/Identity_23*
dtype0
V
save_1/Identity_24Identitysave_1/RestoreV2:23*
T0*
_output_shapes
:
q
save_1/AssignVariableOp_23AssignVariableOp pi/actor/actor/dense_3/bias/Adamsave_1/Identity_24*
dtype0
V
save_1/Identity_25Identitysave_1/RestoreV2:24*
_output_shapes
:*
T0
s
save_1/AssignVariableOp_24AssignVariableOp"pi/actor/actor/dense_3/bias/Adam_1save_1/Identity_25*
dtype0
V
save_1/Identity_26Identitysave_1/RestoreV2:25*
T0*
_output_shapes
:
n
save_1/AssignVariableOp_25AssignVariableOppi/actor/actor/dense_3/kernelsave_1/Identity_26*
dtype0
V
save_1/Identity_27Identitysave_1/RestoreV2:26*
_output_shapes
:*
T0
s
save_1/AssignVariableOp_26AssignVariableOp"pi/actor/actor/dense_3/kernel/Adamsave_1/Identity_27*
dtype0
V
save_1/Identity_28Identitysave_1/RestoreV2:27*
T0*
_output_shapes
:
u
save_1/AssignVariableOp_27AssignVariableOp$pi/actor/actor/dense_3/kernel/Adam_1save_1/Identity_28*
dtype0
V
save_1/Identity_29Identitysave_1/RestoreV2:28*
_output_shapes
:*
T0
m
save_1/AssignVariableOp_28AssignVariableOpv/critic/critic/dense_4/biassave_1/Identity_29*
dtype0
V
save_1/Identity_30Identitysave_1/RestoreV2:29*
T0*
_output_shapes
:
r
save_1/AssignVariableOp_29AssignVariableOp!v/critic/critic/dense_4/bias/Adamsave_1/Identity_30*
dtype0
V
save_1/Identity_31Identitysave_1/RestoreV2:30*
T0*
_output_shapes
:
t
save_1/AssignVariableOp_30AssignVariableOp#v/critic/critic/dense_4/bias/Adam_1save_1/Identity_31*
dtype0
V
save_1/Identity_32Identitysave_1/RestoreV2:31*
_output_shapes
:*
T0
o
save_1/AssignVariableOp_31AssignVariableOpv/critic/critic/dense_4/kernelsave_1/Identity_32*
dtype0
V
save_1/Identity_33Identitysave_1/RestoreV2:32*
T0*
_output_shapes
:
t
save_1/AssignVariableOp_32AssignVariableOp#v/critic/critic/dense_4/kernel/Adamsave_1/Identity_33*
dtype0
V
save_1/Identity_34Identitysave_1/RestoreV2:33*
T0*
_output_shapes
:
v
save_1/AssignVariableOp_33AssignVariableOp%v/critic/critic/dense_4/kernel/Adam_1save_1/Identity_34*
dtype0
V
save_1/Identity_35Identitysave_1/RestoreV2:34*
_output_shapes
:*
T0
m
save_1/AssignVariableOp_34AssignVariableOpv/critic/critic/dense_5/biassave_1/Identity_35*
dtype0
V
save_1/Identity_36Identitysave_1/RestoreV2:35*
T0*
_output_shapes
:
r
save_1/AssignVariableOp_35AssignVariableOp!v/critic/critic/dense_5/bias/Adamsave_1/Identity_36*
dtype0
V
save_1/Identity_37Identitysave_1/RestoreV2:36*
T0*
_output_shapes
:
t
save_1/AssignVariableOp_36AssignVariableOp#v/critic/critic/dense_5/bias/Adam_1save_1/Identity_37*
dtype0
V
save_1/Identity_38Identitysave_1/RestoreV2:37*
_output_shapes
:*
T0
o
save_1/AssignVariableOp_37AssignVariableOpv/critic/critic/dense_5/kernelsave_1/Identity_38*
dtype0
V
save_1/Identity_39Identitysave_1/RestoreV2:38*
T0*
_output_shapes
:
t
save_1/AssignVariableOp_38AssignVariableOp#v/critic/critic/dense_5/kernel/Adamsave_1/Identity_39*
dtype0
V
save_1/Identity_40Identitysave_1/RestoreV2:39*
_output_shapes
:*
T0
v
save_1/AssignVariableOp_39AssignVariableOp%v/critic/critic/dense_5/kernel/Adam_1save_1/Identity_40*
dtype0
V
save_1/Identity_41Identitysave_1/RestoreV2:40*
T0*
_output_shapes
:
m
save_1/AssignVariableOp_40AssignVariableOpv/critic/critic/dense_6/biassave_1/Identity_41*
dtype0
V
save_1/Identity_42Identitysave_1/RestoreV2:41*
_output_shapes
:*
T0
r
save_1/AssignVariableOp_41AssignVariableOp!v/critic/critic/dense_6/bias/Adamsave_1/Identity_42*
dtype0
V
save_1/Identity_43Identitysave_1/RestoreV2:42*
T0*
_output_shapes
:
t
save_1/AssignVariableOp_42AssignVariableOp#v/critic/critic/dense_6/bias/Adam_1save_1/Identity_43*
dtype0
V
save_1/Identity_44Identitysave_1/RestoreV2:43*
T0*
_output_shapes
:
o
save_1/AssignVariableOp_43AssignVariableOpv/critic/critic/dense_6/kernelsave_1/Identity_44*
dtype0
V
save_1/Identity_45Identitysave_1/RestoreV2:44*
T0*
_output_shapes
:
t
save_1/AssignVariableOp_44AssignVariableOp#v/critic/critic/dense_6/kernel/Adamsave_1/Identity_45*
dtype0
V
save_1/Identity_46Identitysave_1/RestoreV2:45*
T0*
_output_shapes
:
v
save_1/AssignVariableOp_45AssignVariableOp%v/critic/critic/dense_6/kernel/Adam_1save_1/Identity_46*
dtype0
Ę

save_1/restore_shardNoOp^save_1/AssignVariableOp^save_1/AssignVariableOp_1^save_1/AssignVariableOp_10^save_1/AssignVariableOp_11^save_1/AssignVariableOp_12^save_1/AssignVariableOp_13^save_1/AssignVariableOp_14^save_1/AssignVariableOp_15^save_1/AssignVariableOp_16^save_1/AssignVariableOp_17^save_1/AssignVariableOp_18^save_1/AssignVariableOp_19^save_1/AssignVariableOp_2^save_1/AssignVariableOp_20^save_1/AssignVariableOp_21^save_1/AssignVariableOp_22^save_1/AssignVariableOp_23^save_1/AssignVariableOp_24^save_1/AssignVariableOp_25^save_1/AssignVariableOp_26^save_1/AssignVariableOp_27^save_1/AssignVariableOp_28^save_1/AssignVariableOp_29^save_1/AssignVariableOp_3^save_1/AssignVariableOp_30^save_1/AssignVariableOp_31^save_1/AssignVariableOp_32^save_1/AssignVariableOp_33^save_1/AssignVariableOp_34^save_1/AssignVariableOp_35^save_1/AssignVariableOp_36^save_1/AssignVariableOp_37^save_1/AssignVariableOp_38^save_1/AssignVariableOp_39^save_1/AssignVariableOp_4^save_1/AssignVariableOp_40^save_1/AssignVariableOp_41^save_1/AssignVariableOp_42^save_1/AssignVariableOp_43^save_1/AssignVariableOp_44^save_1/AssignVariableOp_45^save_1/AssignVariableOp_5^save_1/AssignVariableOp_6^save_1/AssignVariableOp_7^save_1/AssignVariableOp_8^save_1/AssignVariableOp_9
1
save_1/restore_allNoOp^save_1/restore_shard
[
save_2/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
dtype0*
_output_shapes
: *
shape: 

save_2/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_2fcdccdac4d44264a51e97eab07b5d00/part*
dtype0
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_2/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_2/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0

save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
¶
save_2/SaveV2/tensor_namesConst*ē
valueŻBŚ.Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/actor/actor/dense/biasBpi/actor/actor/dense/bias/AdamB pi/actor/actor/dense/bias/Adam_1Bpi/actor/actor/dense/kernelB pi/actor/actor/dense/kernel/AdamB"pi/actor/actor/dense/kernel/Adam_1Bpi/actor/actor/dense_1/biasB pi/actor/actor/dense_1/bias/AdamB"pi/actor/actor/dense_1/bias/Adam_1Bpi/actor/actor/dense_1/kernelB"pi/actor/actor/dense_1/kernel/AdamB$pi/actor/actor/dense_1/kernel/Adam_1Bpi/actor/actor/dense_2/biasB pi/actor/actor/dense_2/bias/AdamB"pi/actor/actor/dense_2/bias/Adam_1Bpi/actor/actor/dense_2/kernelB"pi/actor/actor/dense_2/kernel/AdamB$pi/actor/actor/dense_2/kernel/Adam_1Bpi/actor/actor/dense_3/biasB pi/actor/actor/dense_3/bias/AdamB"pi/actor/actor/dense_3/bias/Adam_1Bpi/actor/actor/dense_3/kernelB"pi/actor/actor/dense_3/kernel/AdamB$pi/actor/actor/dense_3/kernel/Adam_1Bv/critic/critic/dense_4/biasB!v/critic/critic/dense_4/bias/AdamB#v/critic/critic/dense_4/bias/Adam_1Bv/critic/critic/dense_4/kernelB#v/critic/critic/dense_4/kernel/AdamB%v/critic/critic/dense_4/kernel/Adam_1Bv/critic/critic/dense_5/biasB!v/critic/critic/dense_5/bias/AdamB#v/critic/critic/dense_5/bias/Adam_1Bv/critic/critic/dense_5/kernelB#v/critic/critic/dense_5/kernel/AdamB%v/critic/critic/dense_5/kernel/Adam_1Bv/critic/critic/dense_6/biasB!v/critic/critic/dense_6/bias/AdamB#v/critic/critic/dense_6/bias/Adam_1Bv/critic/critic/dense_6/kernelB#v/critic/critic/dense_6/kernel/AdamB%v/critic/critic/dense_6/kernel/Adam_1*
dtype0*
_output_shapes
:.
Į
save_2/SaveV2/shape_and_slicesConst*
_output_shapes
:.*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0

save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesbeta1_power/Read/ReadVariableOp!beta1_power_1/Read/ReadVariableOpbeta2_power/Read/ReadVariableOp!beta2_power_1/Read/ReadVariableOp-pi/actor/actor/dense/bias/Read/ReadVariableOp2pi/actor/actor/dense/bias/Adam/Read/ReadVariableOp4pi/actor/actor/dense/bias/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense/kernel/Read/ReadVariableOp4pi/actor/actor/dense/kernel/Adam/Read/ReadVariableOp6pi/actor/actor/dense/kernel/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense_1/bias/Read/ReadVariableOp4pi/actor/actor/dense_1/bias/Adam/Read/ReadVariableOp6pi/actor/actor/dense_1/bias/Adam_1/Read/ReadVariableOp1pi/actor/actor/dense_1/kernel/Read/ReadVariableOp6pi/actor/actor/dense_1/kernel/Adam/Read/ReadVariableOp8pi/actor/actor/dense_1/kernel/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense_2/bias/Read/ReadVariableOp4pi/actor/actor/dense_2/bias/Adam/Read/ReadVariableOp6pi/actor/actor/dense_2/bias/Adam_1/Read/ReadVariableOp1pi/actor/actor/dense_2/kernel/Read/ReadVariableOp6pi/actor/actor/dense_2/kernel/Adam/Read/ReadVariableOp8pi/actor/actor/dense_2/kernel/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense_3/bias/Read/ReadVariableOp4pi/actor/actor/dense_3/bias/Adam/Read/ReadVariableOp6pi/actor/actor/dense_3/bias/Adam_1/Read/ReadVariableOp1pi/actor/actor/dense_3/kernel/Read/ReadVariableOp6pi/actor/actor/dense_3/kernel/Adam/Read/ReadVariableOp8pi/actor/actor/dense_3/kernel/Adam_1/Read/ReadVariableOp0v/critic/critic/dense_4/bias/Read/ReadVariableOp5v/critic/critic/dense_4/bias/Adam/Read/ReadVariableOp7v/critic/critic/dense_4/bias/Adam_1/Read/ReadVariableOp2v/critic/critic/dense_4/kernel/Read/ReadVariableOp7v/critic/critic/dense_4/kernel/Adam/Read/ReadVariableOp9v/critic/critic/dense_4/kernel/Adam_1/Read/ReadVariableOp0v/critic/critic/dense_5/bias/Read/ReadVariableOp5v/critic/critic/dense_5/bias/Adam/Read/ReadVariableOp7v/critic/critic/dense_5/bias/Adam_1/Read/ReadVariableOp2v/critic/critic/dense_5/kernel/Read/ReadVariableOp7v/critic/critic/dense_5/kernel/Adam/Read/ReadVariableOp9v/critic/critic/dense_5/kernel/Adam_1/Read/ReadVariableOp0v/critic/critic/dense_6/bias/Read/ReadVariableOp5v/critic/critic/dense_6/bias/Adam/Read/ReadVariableOp7v/critic/critic/dense_6/bias/Adam_1/Read/ReadVariableOp2v/critic/critic/dense_6/kernel/Read/ReadVariableOp7v/critic/critic/dense_6/kernel/Adam/Read/ReadVariableOp9v/critic/critic/dense_6/kernel/Adam_1/Read/ReadVariableOp*<
dtypes2
02.

save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_2/ShardedFilename
£
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(

save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
_output_shapes
: *
T0
¹
save_2/RestoreV2/tensor_namesConst*
_output_shapes
:.*ē
valueŻBŚ.Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/actor/actor/dense/biasBpi/actor/actor/dense/bias/AdamB pi/actor/actor/dense/bias/Adam_1Bpi/actor/actor/dense/kernelB pi/actor/actor/dense/kernel/AdamB"pi/actor/actor/dense/kernel/Adam_1Bpi/actor/actor/dense_1/biasB pi/actor/actor/dense_1/bias/AdamB"pi/actor/actor/dense_1/bias/Adam_1Bpi/actor/actor/dense_1/kernelB"pi/actor/actor/dense_1/kernel/AdamB$pi/actor/actor/dense_1/kernel/Adam_1Bpi/actor/actor/dense_2/biasB pi/actor/actor/dense_2/bias/AdamB"pi/actor/actor/dense_2/bias/Adam_1Bpi/actor/actor/dense_2/kernelB"pi/actor/actor/dense_2/kernel/AdamB$pi/actor/actor/dense_2/kernel/Adam_1Bpi/actor/actor/dense_3/biasB pi/actor/actor/dense_3/bias/AdamB"pi/actor/actor/dense_3/bias/Adam_1Bpi/actor/actor/dense_3/kernelB"pi/actor/actor/dense_3/kernel/AdamB$pi/actor/actor/dense_3/kernel/Adam_1Bv/critic/critic/dense_4/biasB!v/critic/critic/dense_4/bias/AdamB#v/critic/critic/dense_4/bias/Adam_1Bv/critic/critic/dense_4/kernelB#v/critic/critic/dense_4/kernel/AdamB%v/critic/critic/dense_4/kernel/Adam_1Bv/critic/critic/dense_5/biasB!v/critic/critic/dense_5/bias/AdamB#v/critic/critic/dense_5/bias/Adam_1Bv/critic/critic/dense_5/kernelB#v/critic/critic/dense_5/kernel/AdamB%v/critic/critic/dense_5/kernel/Adam_1Bv/critic/critic/dense_6/biasB!v/critic/critic/dense_6/bias/AdamB#v/critic/critic/dense_6/bias/Adam_1Bv/critic/critic/dense_6/kernelB#v/critic/critic/dense_6/kernel/AdamB%v/critic/critic/dense_6/kernel/Adam_1*
dtype0
Ä
!save_2/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:.*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ü
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*Ī
_output_shapes»
ø::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.
R
save_2/Identity_1Identitysave_2/RestoreV2*
T0*
_output_shapes
:
X
save_2/AssignVariableOpAssignVariableOpbeta1_powersave_2/Identity_1*
dtype0
T
save_2/Identity_2Identitysave_2/RestoreV2:1*
T0*
_output_shapes
:
\
save_2/AssignVariableOp_1AssignVariableOpbeta1_power_1save_2/Identity_2*
dtype0
T
save_2/Identity_3Identitysave_2/RestoreV2:2*
T0*
_output_shapes
:
Z
save_2/AssignVariableOp_2AssignVariableOpbeta2_powersave_2/Identity_3*
dtype0
T
save_2/Identity_4Identitysave_2/RestoreV2:3*
T0*
_output_shapes
:
\
save_2/AssignVariableOp_3AssignVariableOpbeta2_power_1save_2/Identity_4*
dtype0
T
save_2/Identity_5Identitysave_2/RestoreV2:4*
_output_shapes
:*
T0
h
save_2/AssignVariableOp_4AssignVariableOppi/actor/actor/dense/biassave_2/Identity_5*
dtype0
T
save_2/Identity_6Identitysave_2/RestoreV2:5*
T0*
_output_shapes
:
m
save_2/AssignVariableOp_5AssignVariableOppi/actor/actor/dense/bias/Adamsave_2/Identity_6*
dtype0
T
save_2/Identity_7Identitysave_2/RestoreV2:6*
_output_shapes
:*
T0
o
save_2/AssignVariableOp_6AssignVariableOp pi/actor/actor/dense/bias/Adam_1save_2/Identity_7*
dtype0
T
save_2/Identity_8Identitysave_2/RestoreV2:7*
T0*
_output_shapes
:
j
save_2/AssignVariableOp_7AssignVariableOppi/actor/actor/dense/kernelsave_2/Identity_8*
dtype0
T
save_2/Identity_9Identitysave_2/RestoreV2:8*
T0*
_output_shapes
:
o
save_2/AssignVariableOp_8AssignVariableOp pi/actor/actor/dense/kernel/Adamsave_2/Identity_9*
dtype0
U
save_2/Identity_10Identitysave_2/RestoreV2:9*
T0*
_output_shapes
:
r
save_2/AssignVariableOp_9AssignVariableOp"pi/actor/actor/dense/kernel/Adam_1save_2/Identity_10*
dtype0
V
save_2/Identity_11Identitysave_2/RestoreV2:10*
T0*
_output_shapes
:
l
save_2/AssignVariableOp_10AssignVariableOppi/actor/actor/dense_1/biassave_2/Identity_11*
dtype0
V
save_2/Identity_12Identitysave_2/RestoreV2:11*
_output_shapes
:*
T0
q
save_2/AssignVariableOp_11AssignVariableOp pi/actor/actor/dense_1/bias/Adamsave_2/Identity_12*
dtype0
V
save_2/Identity_13Identitysave_2/RestoreV2:12*
T0*
_output_shapes
:
s
save_2/AssignVariableOp_12AssignVariableOp"pi/actor/actor/dense_1/bias/Adam_1save_2/Identity_13*
dtype0
V
save_2/Identity_14Identitysave_2/RestoreV2:13*
_output_shapes
:*
T0
n
save_2/AssignVariableOp_13AssignVariableOppi/actor/actor/dense_1/kernelsave_2/Identity_14*
dtype0
V
save_2/Identity_15Identitysave_2/RestoreV2:14*
_output_shapes
:*
T0
s
save_2/AssignVariableOp_14AssignVariableOp"pi/actor/actor/dense_1/kernel/Adamsave_2/Identity_15*
dtype0
V
save_2/Identity_16Identitysave_2/RestoreV2:15*
T0*
_output_shapes
:
u
save_2/AssignVariableOp_15AssignVariableOp$pi/actor/actor/dense_1/kernel/Adam_1save_2/Identity_16*
dtype0
V
save_2/Identity_17Identitysave_2/RestoreV2:16*
T0*
_output_shapes
:
l
save_2/AssignVariableOp_16AssignVariableOppi/actor/actor/dense_2/biassave_2/Identity_17*
dtype0
V
save_2/Identity_18Identitysave_2/RestoreV2:17*
_output_shapes
:*
T0
q
save_2/AssignVariableOp_17AssignVariableOp pi/actor/actor/dense_2/bias/Adamsave_2/Identity_18*
dtype0
V
save_2/Identity_19Identitysave_2/RestoreV2:18*
T0*
_output_shapes
:
s
save_2/AssignVariableOp_18AssignVariableOp"pi/actor/actor/dense_2/bias/Adam_1save_2/Identity_19*
dtype0
V
save_2/Identity_20Identitysave_2/RestoreV2:19*
T0*
_output_shapes
:
n
save_2/AssignVariableOp_19AssignVariableOppi/actor/actor/dense_2/kernelsave_2/Identity_20*
dtype0
V
save_2/Identity_21Identitysave_2/RestoreV2:20*
T0*
_output_shapes
:
s
save_2/AssignVariableOp_20AssignVariableOp"pi/actor/actor/dense_2/kernel/Adamsave_2/Identity_21*
dtype0
V
save_2/Identity_22Identitysave_2/RestoreV2:21*
T0*
_output_shapes
:
u
save_2/AssignVariableOp_21AssignVariableOp$pi/actor/actor/dense_2/kernel/Adam_1save_2/Identity_22*
dtype0
V
save_2/Identity_23Identitysave_2/RestoreV2:22*
T0*
_output_shapes
:
l
save_2/AssignVariableOp_22AssignVariableOppi/actor/actor/dense_3/biassave_2/Identity_23*
dtype0
V
save_2/Identity_24Identitysave_2/RestoreV2:23*
_output_shapes
:*
T0
q
save_2/AssignVariableOp_23AssignVariableOp pi/actor/actor/dense_3/bias/Adamsave_2/Identity_24*
dtype0
V
save_2/Identity_25Identitysave_2/RestoreV2:24*
T0*
_output_shapes
:
s
save_2/AssignVariableOp_24AssignVariableOp"pi/actor/actor/dense_3/bias/Adam_1save_2/Identity_25*
dtype0
V
save_2/Identity_26Identitysave_2/RestoreV2:25*
_output_shapes
:*
T0
n
save_2/AssignVariableOp_25AssignVariableOppi/actor/actor/dense_3/kernelsave_2/Identity_26*
dtype0
V
save_2/Identity_27Identitysave_2/RestoreV2:26*
_output_shapes
:*
T0
s
save_2/AssignVariableOp_26AssignVariableOp"pi/actor/actor/dense_3/kernel/Adamsave_2/Identity_27*
dtype0
V
save_2/Identity_28Identitysave_2/RestoreV2:27*
_output_shapes
:*
T0
u
save_2/AssignVariableOp_27AssignVariableOp$pi/actor/actor/dense_3/kernel/Adam_1save_2/Identity_28*
dtype0
V
save_2/Identity_29Identitysave_2/RestoreV2:28*
T0*
_output_shapes
:
m
save_2/AssignVariableOp_28AssignVariableOpv/critic/critic/dense_4/biassave_2/Identity_29*
dtype0
V
save_2/Identity_30Identitysave_2/RestoreV2:29*
T0*
_output_shapes
:
r
save_2/AssignVariableOp_29AssignVariableOp!v/critic/critic/dense_4/bias/Adamsave_2/Identity_30*
dtype0
V
save_2/Identity_31Identitysave_2/RestoreV2:30*
T0*
_output_shapes
:
t
save_2/AssignVariableOp_30AssignVariableOp#v/critic/critic/dense_4/bias/Adam_1save_2/Identity_31*
dtype0
V
save_2/Identity_32Identitysave_2/RestoreV2:31*
_output_shapes
:*
T0
o
save_2/AssignVariableOp_31AssignVariableOpv/critic/critic/dense_4/kernelsave_2/Identity_32*
dtype0
V
save_2/Identity_33Identitysave_2/RestoreV2:32*
T0*
_output_shapes
:
t
save_2/AssignVariableOp_32AssignVariableOp#v/critic/critic/dense_4/kernel/Adamsave_2/Identity_33*
dtype0
V
save_2/Identity_34Identitysave_2/RestoreV2:33*
T0*
_output_shapes
:
v
save_2/AssignVariableOp_33AssignVariableOp%v/critic/critic/dense_4/kernel/Adam_1save_2/Identity_34*
dtype0
V
save_2/Identity_35Identitysave_2/RestoreV2:34*
T0*
_output_shapes
:
m
save_2/AssignVariableOp_34AssignVariableOpv/critic/critic/dense_5/biassave_2/Identity_35*
dtype0
V
save_2/Identity_36Identitysave_2/RestoreV2:35*
_output_shapes
:*
T0
r
save_2/AssignVariableOp_35AssignVariableOp!v/critic/critic/dense_5/bias/Adamsave_2/Identity_36*
dtype0
V
save_2/Identity_37Identitysave_2/RestoreV2:36*
T0*
_output_shapes
:
t
save_2/AssignVariableOp_36AssignVariableOp#v/critic/critic/dense_5/bias/Adam_1save_2/Identity_37*
dtype0
V
save_2/Identity_38Identitysave_2/RestoreV2:37*
T0*
_output_shapes
:
o
save_2/AssignVariableOp_37AssignVariableOpv/critic/critic/dense_5/kernelsave_2/Identity_38*
dtype0
V
save_2/Identity_39Identitysave_2/RestoreV2:38*
T0*
_output_shapes
:
t
save_2/AssignVariableOp_38AssignVariableOp#v/critic/critic/dense_5/kernel/Adamsave_2/Identity_39*
dtype0
V
save_2/Identity_40Identitysave_2/RestoreV2:39*
T0*
_output_shapes
:
v
save_2/AssignVariableOp_39AssignVariableOp%v/critic/critic/dense_5/kernel/Adam_1save_2/Identity_40*
dtype0
V
save_2/Identity_41Identitysave_2/RestoreV2:40*
T0*
_output_shapes
:
m
save_2/AssignVariableOp_40AssignVariableOpv/critic/critic/dense_6/biassave_2/Identity_41*
dtype0
V
save_2/Identity_42Identitysave_2/RestoreV2:41*
T0*
_output_shapes
:
r
save_2/AssignVariableOp_41AssignVariableOp!v/critic/critic/dense_6/bias/Adamsave_2/Identity_42*
dtype0
V
save_2/Identity_43Identitysave_2/RestoreV2:42*
T0*
_output_shapes
:
t
save_2/AssignVariableOp_42AssignVariableOp#v/critic/critic/dense_6/bias/Adam_1save_2/Identity_43*
dtype0
V
save_2/Identity_44Identitysave_2/RestoreV2:43*
T0*
_output_shapes
:
o
save_2/AssignVariableOp_43AssignVariableOpv/critic/critic/dense_6/kernelsave_2/Identity_44*
dtype0
V
save_2/Identity_45Identitysave_2/RestoreV2:44*
_output_shapes
:*
T0
t
save_2/AssignVariableOp_44AssignVariableOp#v/critic/critic/dense_6/kernel/Adamsave_2/Identity_45*
dtype0
V
save_2/Identity_46Identitysave_2/RestoreV2:45*
_output_shapes
:*
T0
v
save_2/AssignVariableOp_45AssignVariableOp%v/critic/critic/dense_6/kernel/Adam_1save_2/Identity_46*
dtype0
Ę

save_2/restore_shardNoOp^save_2/AssignVariableOp^save_2/AssignVariableOp_1^save_2/AssignVariableOp_10^save_2/AssignVariableOp_11^save_2/AssignVariableOp_12^save_2/AssignVariableOp_13^save_2/AssignVariableOp_14^save_2/AssignVariableOp_15^save_2/AssignVariableOp_16^save_2/AssignVariableOp_17^save_2/AssignVariableOp_18^save_2/AssignVariableOp_19^save_2/AssignVariableOp_2^save_2/AssignVariableOp_20^save_2/AssignVariableOp_21^save_2/AssignVariableOp_22^save_2/AssignVariableOp_23^save_2/AssignVariableOp_24^save_2/AssignVariableOp_25^save_2/AssignVariableOp_26^save_2/AssignVariableOp_27^save_2/AssignVariableOp_28^save_2/AssignVariableOp_29^save_2/AssignVariableOp_3^save_2/AssignVariableOp_30^save_2/AssignVariableOp_31^save_2/AssignVariableOp_32^save_2/AssignVariableOp_33^save_2/AssignVariableOp_34^save_2/AssignVariableOp_35^save_2/AssignVariableOp_36^save_2/AssignVariableOp_37^save_2/AssignVariableOp_38^save_2/AssignVariableOp_39^save_2/AssignVariableOp_4^save_2/AssignVariableOp_40^save_2/AssignVariableOp_41^save_2/AssignVariableOp_42^save_2/AssignVariableOp_43^save_2/AssignVariableOp_44^save_2/AssignVariableOp_45^save_2/AssignVariableOp_5^save_2/AssignVariableOp_6^save_2/AssignVariableOp_7^save_2/AssignVariableOp_8^save_2/AssignVariableOp_9
1
save_2/restore_allNoOp^save_2/restore_shard
[
save_3/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_3/filenamePlaceholderWithDefaultsave_3/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_3/ConstPlaceholderWithDefaultsave_3/filename*
dtype0*
_output_shapes
: *
shape: 

save_3/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_79e01c7d973b44709541c5ea0057ec63/part
{
save_3/StringJoin
StringJoinsave_3/Constsave_3/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_3/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_3/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
¶
save_3/SaveV2/tensor_namesConst*ē
valueŻBŚ.Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/actor/actor/dense/biasBpi/actor/actor/dense/bias/AdamB pi/actor/actor/dense/bias/Adam_1Bpi/actor/actor/dense/kernelB pi/actor/actor/dense/kernel/AdamB"pi/actor/actor/dense/kernel/Adam_1Bpi/actor/actor/dense_1/biasB pi/actor/actor/dense_1/bias/AdamB"pi/actor/actor/dense_1/bias/Adam_1Bpi/actor/actor/dense_1/kernelB"pi/actor/actor/dense_1/kernel/AdamB$pi/actor/actor/dense_1/kernel/Adam_1Bpi/actor/actor/dense_2/biasB pi/actor/actor/dense_2/bias/AdamB"pi/actor/actor/dense_2/bias/Adam_1Bpi/actor/actor/dense_2/kernelB"pi/actor/actor/dense_2/kernel/AdamB$pi/actor/actor/dense_2/kernel/Adam_1Bpi/actor/actor/dense_3/biasB pi/actor/actor/dense_3/bias/AdamB"pi/actor/actor/dense_3/bias/Adam_1Bpi/actor/actor/dense_3/kernelB"pi/actor/actor/dense_3/kernel/AdamB$pi/actor/actor/dense_3/kernel/Adam_1Bv/critic/critic/dense_4/biasB!v/critic/critic/dense_4/bias/AdamB#v/critic/critic/dense_4/bias/Adam_1Bv/critic/critic/dense_4/kernelB#v/critic/critic/dense_4/kernel/AdamB%v/critic/critic/dense_4/kernel/Adam_1Bv/critic/critic/dense_5/biasB!v/critic/critic/dense_5/bias/AdamB#v/critic/critic/dense_5/bias/Adam_1Bv/critic/critic/dense_5/kernelB#v/critic/critic/dense_5/kernel/AdamB%v/critic/critic/dense_5/kernel/Adam_1Bv/critic/critic/dense_6/biasB!v/critic/critic/dense_6/bias/AdamB#v/critic/critic/dense_6/bias/Adam_1Bv/critic/critic/dense_6/kernelB#v/critic/critic/dense_6/kernel/AdamB%v/critic/critic/dense_6/kernel/Adam_1*
dtype0*
_output_shapes
:.
Į
save_3/SaveV2/shape_and_slicesConst*
_output_shapes
:.*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0

save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesbeta1_power/Read/ReadVariableOp!beta1_power_1/Read/ReadVariableOpbeta2_power/Read/ReadVariableOp!beta2_power_1/Read/ReadVariableOp-pi/actor/actor/dense/bias/Read/ReadVariableOp2pi/actor/actor/dense/bias/Adam/Read/ReadVariableOp4pi/actor/actor/dense/bias/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense/kernel/Read/ReadVariableOp4pi/actor/actor/dense/kernel/Adam/Read/ReadVariableOp6pi/actor/actor/dense/kernel/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense_1/bias/Read/ReadVariableOp4pi/actor/actor/dense_1/bias/Adam/Read/ReadVariableOp6pi/actor/actor/dense_1/bias/Adam_1/Read/ReadVariableOp1pi/actor/actor/dense_1/kernel/Read/ReadVariableOp6pi/actor/actor/dense_1/kernel/Adam/Read/ReadVariableOp8pi/actor/actor/dense_1/kernel/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense_2/bias/Read/ReadVariableOp4pi/actor/actor/dense_2/bias/Adam/Read/ReadVariableOp6pi/actor/actor/dense_2/bias/Adam_1/Read/ReadVariableOp1pi/actor/actor/dense_2/kernel/Read/ReadVariableOp6pi/actor/actor/dense_2/kernel/Adam/Read/ReadVariableOp8pi/actor/actor/dense_2/kernel/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense_3/bias/Read/ReadVariableOp4pi/actor/actor/dense_3/bias/Adam/Read/ReadVariableOp6pi/actor/actor/dense_3/bias/Adam_1/Read/ReadVariableOp1pi/actor/actor/dense_3/kernel/Read/ReadVariableOp6pi/actor/actor/dense_3/kernel/Adam/Read/ReadVariableOp8pi/actor/actor/dense_3/kernel/Adam_1/Read/ReadVariableOp0v/critic/critic/dense_4/bias/Read/ReadVariableOp5v/critic/critic/dense_4/bias/Adam/Read/ReadVariableOp7v/critic/critic/dense_4/bias/Adam_1/Read/ReadVariableOp2v/critic/critic/dense_4/kernel/Read/ReadVariableOp7v/critic/critic/dense_4/kernel/Adam/Read/ReadVariableOp9v/critic/critic/dense_4/kernel/Adam_1/Read/ReadVariableOp0v/critic/critic/dense_5/bias/Read/ReadVariableOp5v/critic/critic/dense_5/bias/Adam/Read/ReadVariableOp7v/critic/critic/dense_5/bias/Adam_1/Read/ReadVariableOp2v/critic/critic/dense_5/kernel/Read/ReadVariableOp7v/critic/critic/dense_5/kernel/Adam/Read/ReadVariableOp9v/critic/critic/dense_5/kernel/Adam_1/Read/ReadVariableOp0v/critic/critic/dense_6/bias/Read/ReadVariableOp5v/critic/critic/dense_6/bias/Adam/Read/ReadVariableOp7v/critic/critic/dense_6/bias/Adam_1/Read/ReadVariableOp2v/critic/critic/dense_6/kernel/Read/ReadVariableOp7v/critic/critic/dense_6/kernel/Adam/Read/ReadVariableOp9v/critic/critic/dense_6/kernel/Adam_1/Read/ReadVariableOp*<
dtypes2
02.

save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*
T0*)
_class
loc:@save_3/ShardedFilename*
_output_shapes
: 
£
-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilename^save_3/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_3/MergeV2CheckpointsMergeV2Checkpoints-save_3/MergeV2Checkpoints/checkpoint_prefixessave_3/Const*
delete_old_dirs(

save_3/IdentityIdentitysave_3/Const^save_3/MergeV2Checkpoints^save_3/control_dependency*
T0*
_output_shapes
: 
¹
save_3/RestoreV2/tensor_namesConst*ē
valueŻBŚ.Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/actor/actor/dense/biasBpi/actor/actor/dense/bias/AdamB pi/actor/actor/dense/bias/Adam_1Bpi/actor/actor/dense/kernelB pi/actor/actor/dense/kernel/AdamB"pi/actor/actor/dense/kernel/Adam_1Bpi/actor/actor/dense_1/biasB pi/actor/actor/dense_1/bias/AdamB"pi/actor/actor/dense_1/bias/Adam_1Bpi/actor/actor/dense_1/kernelB"pi/actor/actor/dense_1/kernel/AdamB$pi/actor/actor/dense_1/kernel/Adam_1Bpi/actor/actor/dense_2/biasB pi/actor/actor/dense_2/bias/AdamB"pi/actor/actor/dense_2/bias/Adam_1Bpi/actor/actor/dense_2/kernelB"pi/actor/actor/dense_2/kernel/AdamB$pi/actor/actor/dense_2/kernel/Adam_1Bpi/actor/actor/dense_3/biasB pi/actor/actor/dense_3/bias/AdamB"pi/actor/actor/dense_3/bias/Adam_1Bpi/actor/actor/dense_3/kernelB"pi/actor/actor/dense_3/kernel/AdamB$pi/actor/actor/dense_3/kernel/Adam_1Bv/critic/critic/dense_4/biasB!v/critic/critic/dense_4/bias/AdamB#v/critic/critic/dense_4/bias/Adam_1Bv/critic/critic/dense_4/kernelB#v/critic/critic/dense_4/kernel/AdamB%v/critic/critic/dense_4/kernel/Adam_1Bv/critic/critic/dense_5/biasB!v/critic/critic/dense_5/bias/AdamB#v/critic/critic/dense_5/bias/Adam_1Bv/critic/critic/dense_5/kernelB#v/critic/critic/dense_5/kernel/AdamB%v/critic/critic/dense_5/kernel/Adam_1Bv/critic/critic/dense_6/biasB!v/critic/critic/dense_6/bias/AdamB#v/critic/critic/dense_6/bias/Adam_1Bv/critic/critic/dense_6/kernelB#v/critic/critic/dense_6/kernel/AdamB%v/critic/critic/dense_6/kernel/Adam_1*
dtype0*
_output_shapes
:.
Ä
!save_3/RestoreV2/shape_and_slicesConst*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:.
ü
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*Ī
_output_shapes»
ø::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.
R
save_3/Identity_1Identitysave_3/RestoreV2*
T0*
_output_shapes
:
X
save_3/AssignVariableOpAssignVariableOpbeta1_powersave_3/Identity_1*
dtype0
T
save_3/Identity_2Identitysave_3/RestoreV2:1*
_output_shapes
:*
T0
\
save_3/AssignVariableOp_1AssignVariableOpbeta1_power_1save_3/Identity_2*
dtype0
T
save_3/Identity_3Identitysave_3/RestoreV2:2*
T0*
_output_shapes
:
Z
save_3/AssignVariableOp_2AssignVariableOpbeta2_powersave_3/Identity_3*
dtype0
T
save_3/Identity_4Identitysave_3/RestoreV2:3*
_output_shapes
:*
T0
\
save_3/AssignVariableOp_3AssignVariableOpbeta2_power_1save_3/Identity_4*
dtype0
T
save_3/Identity_5Identitysave_3/RestoreV2:4*
T0*
_output_shapes
:
h
save_3/AssignVariableOp_4AssignVariableOppi/actor/actor/dense/biassave_3/Identity_5*
dtype0
T
save_3/Identity_6Identitysave_3/RestoreV2:5*
T0*
_output_shapes
:
m
save_3/AssignVariableOp_5AssignVariableOppi/actor/actor/dense/bias/Adamsave_3/Identity_6*
dtype0
T
save_3/Identity_7Identitysave_3/RestoreV2:6*
_output_shapes
:*
T0
o
save_3/AssignVariableOp_6AssignVariableOp pi/actor/actor/dense/bias/Adam_1save_3/Identity_7*
dtype0
T
save_3/Identity_8Identitysave_3/RestoreV2:7*
T0*
_output_shapes
:
j
save_3/AssignVariableOp_7AssignVariableOppi/actor/actor/dense/kernelsave_3/Identity_8*
dtype0
T
save_3/Identity_9Identitysave_3/RestoreV2:8*
T0*
_output_shapes
:
o
save_3/AssignVariableOp_8AssignVariableOp pi/actor/actor/dense/kernel/Adamsave_3/Identity_9*
dtype0
U
save_3/Identity_10Identitysave_3/RestoreV2:9*
T0*
_output_shapes
:
r
save_3/AssignVariableOp_9AssignVariableOp"pi/actor/actor/dense/kernel/Adam_1save_3/Identity_10*
dtype0
V
save_3/Identity_11Identitysave_3/RestoreV2:10*
T0*
_output_shapes
:
l
save_3/AssignVariableOp_10AssignVariableOppi/actor/actor/dense_1/biassave_3/Identity_11*
dtype0
V
save_3/Identity_12Identitysave_3/RestoreV2:11*
_output_shapes
:*
T0
q
save_3/AssignVariableOp_11AssignVariableOp pi/actor/actor/dense_1/bias/Adamsave_3/Identity_12*
dtype0
V
save_3/Identity_13Identitysave_3/RestoreV2:12*
_output_shapes
:*
T0
s
save_3/AssignVariableOp_12AssignVariableOp"pi/actor/actor/dense_1/bias/Adam_1save_3/Identity_13*
dtype0
V
save_3/Identity_14Identitysave_3/RestoreV2:13*
T0*
_output_shapes
:
n
save_3/AssignVariableOp_13AssignVariableOppi/actor/actor/dense_1/kernelsave_3/Identity_14*
dtype0
V
save_3/Identity_15Identitysave_3/RestoreV2:14*
T0*
_output_shapes
:
s
save_3/AssignVariableOp_14AssignVariableOp"pi/actor/actor/dense_1/kernel/Adamsave_3/Identity_15*
dtype0
V
save_3/Identity_16Identitysave_3/RestoreV2:15*
T0*
_output_shapes
:
u
save_3/AssignVariableOp_15AssignVariableOp$pi/actor/actor/dense_1/kernel/Adam_1save_3/Identity_16*
dtype0
V
save_3/Identity_17Identitysave_3/RestoreV2:16*
T0*
_output_shapes
:
l
save_3/AssignVariableOp_16AssignVariableOppi/actor/actor/dense_2/biassave_3/Identity_17*
dtype0
V
save_3/Identity_18Identitysave_3/RestoreV2:17*
_output_shapes
:*
T0
q
save_3/AssignVariableOp_17AssignVariableOp pi/actor/actor/dense_2/bias/Adamsave_3/Identity_18*
dtype0
V
save_3/Identity_19Identitysave_3/RestoreV2:18*
T0*
_output_shapes
:
s
save_3/AssignVariableOp_18AssignVariableOp"pi/actor/actor/dense_2/bias/Adam_1save_3/Identity_19*
dtype0
V
save_3/Identity_20Identitysave_3/RestoreV2:19*
T0*
_output_shapes
:
n
save_3/AssignVariableOp_19AssignVariableOppi/actor/actor/dense_2/kernelsave_3/Identity_20*
dtype0
V
save_3/Identity_21Identitysave_3/RestoreV2:20*
_output_shapes
:*
T0
s
save_3/AssignVariableOp_20AssignVariableOp"pi/actor/actor/dense_2/kernel/Adamsave_3/Identity_21*
dtype0
V
save_3/Identity_22Identitysave_3/RestoreV2:21*
T0*
_output_shapes
:
u
save_3/AssignVariableOp_21AssignVariableOp$pi/actor/actor/dense_2/kernel/Adam_1save_3/Identity_22*
dtype0
V
save_3/Identity_23Identitysave_3/RestoreV2:22*
T0*
_output_shapes
:
l
save_3/AssignVariableOp_22AssignVariableOppi/actor/actor/dense_3/biassave_3/Identity_23*
dtype0
V
save_3/Identity_24Identitysave_3/RestoreV2:23*
T0*
_output_shapes
:
q
save_3/AssignVariableOp_23AssignVariableOp pi/actor/actor/dense_3/bias/Adamsave_3/Identity_24*
dtype0
V
save_3/Identity_25Identitysave_3/RestoreV2:24*
_output_shapes
:*
T0
s
save_3/AssignVariableOp_24AssignVariableOp"pi/actor/actor/dense_3/bias/Adam_1save_3/Identity_25*
dtype0
V
save_3/Identity_26Identitysave_3/RestoreV2:25*
_output_shapes
:*
T0
n
save_3/AssignVariableOp_25AssignVariableOppi/actor/actor/dense_3/kernelsave_3/Identity_26*
dtype0
V
save_3/Identity_27Identitysave_3/RestoreV2:26*
T0*
_output_shapes
:
s
save_3/AssignVariableOp_26AssignVariableOp"pi/actor/actor/dense_3/kernel/Adamsave_3/Identity_27*
dtype0
V
save_3/Identity_28Identitysave_3/RestoreV2:27*
T0*
_output_shapes
:
u
save_3/AssignVariableOp_27AssignVariableOp$pi/actor/actor/dense_3/kernel/Adam_1save_3/Identity_28*
dtype0
V
save_3/Identity_29Identitysave_3/RestoreV2:28*
T0*
_output_shapes
:
m
save_3/AssignVariableOp_28AssignVariableOpv/critic/critic/dense_4/biassave_3/Identity_29*
dtype0
V
save_3/Identity_30Identitysave_3/RestoreV2:29*
T0*
_output_shapes
:
r
save_3/AssignVariableOp_29AssignVariableOp!v/critic/critic/dense_4/bias/Adamsave_3/Identity_30*
dtype0
V
save_3/Identity_31Identitysave_3/RestoreV2:30*
T0*
_output_shapes
:
t
save_3/AssignVariableOp_30AssignVariableOp#v/critic/critic/dense_4/bias/Adam_1save_3/Identity_31*
dtype0
V
save_3/Identity_32Identitysave_3/RestoreV2:31*
T0*
_output_shapes
:
o
save_3/AssignVariableOp_31AssignVariableOpv/critic/critic/dense_4/kernelsave_3/Identity_32*
dtype0
V
save_3/Identity_33Identitysave_3/RestoreV2:32*
T0*
_output_shapes
:
t
save_3/AssignVariableOp_32AssignVariableOp#v/critic/critic/dense_4/kernel/Adamsave_3/Identity_33*
dtype0
V
save_3/Identity_34Identitysave_3/RestoreV2:33*
_output_shapes
:*
T0
v
save_3/AssignVariableOp_33AssignVariableOp%v/critic/critic/dense_4/kernel/Adam_1save_3/Identity_34*
dtype0
V
save_3/Identity_35Identitysave_3/RestoreV2:34*
T0*
_output_shapes
:
m
save_3/AssignVariableOp_34AssignVariableOpv/critic/critic/dense_5/biassave_3/Identity_35*
dtype0
V
save_3/Identity_36Identitysave_3/RestoreV2:35*
_output_shapes
:*
T0
r
save_3/AssignVariableOp_35AssignVariableOp!v/critic/critic/dense_5/bias/Adamsave_3/Identity_36*
dtype0
V
save_3/Identity_37Identitysave_3/RestoreV2:36*
_output_shapes
:*
T0
t
save_3/AssignVariableOp_36AssignVariableOp#v/critic/critic/dense_5/bias/Adam_1save_3/Identity_37*
dtype0
V
save_3/Identity_38Identitysave_3/RestoreV2:37*
T0*
_output_shapes
:
o
save_3/AssignVariableOp_37AssignVariableOpv/critic/critic/dense_5/kernelsave_3/Identity_38*
dtype0
V
save_3/Identity_39Identitysave_3/RestoreV2:38*
_output_shapes
:*
T0
t
save_3/AssignVariableOp_38AssignVariableOp#v/critic/critic/dense_5/kernel/Adamsave_3/Identity_39*
dtype0
V
save_3/Identity_40Identitysave_3/RestoreV2:39*
_output_shapes
:*
T0
v
save_3/AssignVariableOp_39AssignVariableOp%v/critic/critic/dense_5/kernel/Adam_1save_3/Identity_40*
dtype0
V
save_3/Identity_41Identitysave_3/RestoreV2:40*
_output_shapes
:*
T0
m
save_3/AssignVariableOp_40AssignVariableOpv/critic/critic/dense_6/biassave_3/Identity_41*
dtype0
V
save_3/Identity_42Identitysave_3/RestoreV2:41*
T0*
_output_shapes
:
r
save_3/AssignVariableOp_41AssignVariableOp!v/critic/critic/dense_6/bias/Adamsave_3/Identity_42*
dtype0
V
save_3/Identity_43Identitysave_3/RestoreV2:42*
T0*
_output_shapes
:
t
save_3/AssignVariableOp_42AssignVariableOp#v/critic/critic/dense_6/bias/Adam_1save_3/Identity_43*
dtype0
V
save_3/Identity_44Identitysave_3/RestoreV2:43*
T0*
_output_shapes
:
o
save_3/AssignVariableOp_43AssignVariableOpv/critic/critic/dense_6/kernelsave_3/Identity_44*
dtype0
V
save_3/Identity_45Identitysave_3/RestoreV2:44*
T0*
_output_shapes
:
t
save_3/AssignVariableOp_44AssignVariableOp#v/critic/critic/dense_6/kernel/Adamsave_3/Identity_45*
dtype0
V
save_3/Identity_46Identitysave_3/RestoreV2:45*
T0*
_output_shapes
:
v
save_3/AssignVariableOp_45AssignVariableOp%v/critic/critic/dense_6/kernel/Adam_1save_3/Identity_46*
dtype0
Ę

save_3/restore_shardNoOp^save_3/AssignVariableOp^save_3/AssignVariableOp_1^save_3/AssignVariableOp_10^save_3/AssignVariableOp_11^save_3/AssignVariableOp_12^save_3/AssignVariableOp_13^save_3/AssignVariableOp_14^save_3/AssignVariableOp_15^save_3/AssignVariableOp_16^save_3/AssignVariableOp_17^save_3/AssignVariableOp_18^save_3/AssignVariableOp_19^save_3/AssignVariableOp_2^save_3/AssignVariableOp_20^save_3/AssignVariableOp_21^save_3/AssignVariableOp_22^save_3/AssignVariableOp_23^save_3/AssignVariableOp_24^save_3/AssignVariableOp_25^save_3/AssignVariableOp_26^save_3/AssignVariableOp_27^save_3/AssignVariableOp_28^save_3/AssignVariableOp_29^save_3/AssignVariableOp_3^save_3/AssignVariableOp_30^save_3/AssignVariableOp_31^save_3/AssignVariableOp_32^save_3/AssignVariableOp_33^save_3/AssignVariableOp_34^save_3/AssignVariableOp_35^save_3/AssignVariableOp_36^save_3/AssignVariableOp_37^save_3/AssignVariableOp_38^save_3/AssignVariableOp_39^save_3/AssignVariableOp_4^save_3/AssignVariableOp_40^save_3/AssignVariableOp_41^save_3/AssignVariableOp_42^save_3/AssignVariableOp_43^save_3/AssignVariableOp_44^save_3/AssignVariableOp_45^save_3/AssignVariableOp_5^save_3/AssignVariableOp_6^save_3/AssignVariableOp_7^save_3/AssignVariableOp_8^save_3/AssignVariableOp_9
1
save_3/restore_allNoOp^save_3/restore_shard
[
save_4/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_4/filenamePlaceholderWithDefaultsave_4/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_4/ConstPlaceholderWithDefaultsave_4/filename*
dtype0*
_output_shapes
: *
shape: 

save_4/StringJoin/inputs_1Const*<
value3B1 B+_temp_daa918e66be948338faa13b90accd41a/part*
dtype0*
_output_shapes
: 
{
save_4/StringJoin
StringJoinsave_4/Constsave_4/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_4/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_4/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_4/ShardedFilenameShardedFilenamesave_4/StringJoinsave_4/ShardedFilename/shardsave_4/num_shards*
_output_shapes
: 
¶
save_4/SaveV2/tensor_namesConst*ē
valueŻBŚ.Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/actor/actor/dense/biasBpi/actor/actor/dense/bias/AdamB pi/actor/actor/dense/bias/Adam_1Bpi/actor/actor/dense/kernelB pi/actor/actor/dense/kernel/AdamB"pi/actor/actor/dense/kernel/Adam_1Bpi/actor/actor/dense_1/biasB pi/actor/actor/dense_1/bias/AdamB"pi/actor/actor/dense_1/bias/Adam_1Bpi/actor/actor/dense_1/kernelB"pi/actor/actor/dense_1/kernel/AdamB$pi/actor/actor/dense_1/kernel/Adam_1Bpi/actor/actor/dense_2/biasB pi/actor/actor/dense_2/bias/AdamB"pi/actor/actor/dense_2/bias/Adam_1Bpi/actor/actor/dense_2/kernelB"pi/actor/actor/dense_2/kernel/AdamB$pi/actor/actor/dense_2/kernel/Adam_1Bpi/actor/actor/dense_3/biasB pi/actor/actor/dense_3/bias/AdamB"pi/actor/actor/dense_3/bias/Adam_1Bpi/actor/actor/dense_3/kernelB"pi/actor/actor/dense_3/kernel/AdamB$pi/actor/actor/dense_3/kernel/Adam_1Bv/critic/critic/dense_4/biasB!v/critic/critic/dense_4/bias/AdamB#v/critic/critic/dense_4/bias/Adam_1Bv/critic/critic/dense_4/kernelB#v/critic/critic/dense_4/kernel/AdamB%v/critic/critic/dense_4/kernel/Adam_1Bv/critic/critic/dense_5/biasB!v/critic/critic/dense_5/bias/AdamB#v/critic/critic/dense_5/bias/Adam_1Bv/critic/critic/dense_5/kernelB#v/critic/critic/dense_5/kernel/AdamB%v/critic/critic/dense_5/kernel/Adam_1Bv/critic/critic/dense_6/biasB!v/critic/critic/dense_6/bias/AdamB#v/critic/critic/dense_6/bias/Adam_1Bv/critic/critic/dense_6/kernelB#v/critic/critic/dense_6/kernel/AdamB%v/critic/critic/dense_6/kernel/Adam_1*
dtype0*
_output_shapes
:.
Į
save_4/SaveV2/shape_and_slicesConst*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:.

save_4/SaveV2SaveV2save_4/ShardedFilenamesave_4/SaveV2/tensor_namessave_4/SaveV2/shape_and_slicesbeta1_power/Read/ReadVariableOp!beta1_power_1/Read/ReadVariableOpbeta2_power/Read/ReadVariableOp!beta2_power_1/Read/ReadVariableOp-pi/actor/actor/dense/bias/Read/ReadVariableOp2pi/actor/actor/dense/bias/Adam/Read/ReadVariableOp4pi/actor/actor/dense/bias/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense/kernel/Read/ReadVariableOp4pi/actor/actor/dense/kernel/Adam/Read/ReadVariableOp6pi/actor/actor/dense/kernel/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense_1/bias/Read/ReadVariableOp4pi/actor/actor/dense_1/bias/Adam/Read/ReadVariableOp6pi/actor/actor/dense_1/bias/Adam_1/Read/ReadVariableOp1pi/actor/actor/dense_1/kernel/Read/ReadVariableOp6pi/actor/actor/dense_1/kernel/Adam/Read/ReadVariableOp8pi/actor/actor/dense_1/kernel/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense_2/bias/Read/ReadVariableOp4pi/actor/actor/dense_2/bias/Adam/Read/ReadVariableOp6pi/actor/actor/dense_2/bias/Adam_1/Read/ReadVariableOp1pi/actor/actor/dense_2/kernel/Read/ReadVariableOp6pi/actor/actor/dense_2/kernel/Adam/Read/ReadVariableOp8pi/actor/actor/dense_2/kernel/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense_3/bias/Read/ReadVariableOp4pi/actor/actor/dense_3/bias/Adam/Read/ReadVariableOp6pi/actor/actor/dense_3/bias/Adam_1/Read/ReadVariableOp1pi/actor/actor/dense_3/kernel/Read/ReadVariableOp6pi/actor/actor/dense_3/kernel/Adam/Read/ReadVariableOp8pi/actor/actor/dense_3/kernel/Adam_1/Read/ReadVariableOp0v/critic/critic/dense_4/bias/Read/ReadVariableOp5v/critic/critic/dense_4/bias/Adam/Read/ReadVariableOp7v/critic/critic/dense_4/bias/Adam_1/Read/ReadVariableOp2v/critic/critic/dense_4/kernel/Read/ReadVariableOp7v/critic/critic/dense_4/kernel/Adam/Read/ReadVariableOp9v/critic/critic/dense_4/kernel/Adam_1/Read/ReadVariableOp0v/critic/critic/dense_5/bias/Read/ReadVariableOp5v/critic/critic/dense_5/bias/Adam/Read/ReadVariableOp7v/critic/critic/dense_5/bias/Adam_1/Read/ReadVariableOp2v/critic/critic/dense_5/kernel/Read/ReadVariableOp7v/critic/critic/dense_5/kernel/Adam/Read/ReadVariableOp9v/critic/critic/dense_5/kernel/Adam_1/Read/ReadVariableOp0v/critic/critic/dense_6/bias/Read/ReadVariableOp5v/critic/critic/dense_6/bias/Adam/Read/ReadVariableOp7v/critic/critic/dense_6/bias/Adam_1/Read/ReadVariableOp2v/critic/critic/dense_6/kernel/Read/ReadVariableOp7v/critic/critic/dense_6/kernel/Adam/Read/ReadVariableOp9v/critic/critic/dense_6/kernel/Adam_1/Read/ReadVariableOp*<
dtypes2
02.

save_4/control_dependencyIdentitysave_4/ShardedFilename^save_4/SaveV2*)
_class
loc:@save_4/ShardedFilename*
_output_shapes
: *
T0
£
-save_4/MergeV2Checkpoints/checkpoint_prefixesPacksave_4/ShardedFilename^save_4/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_4/MergeV2CheckpointsMergeV2Checkpoints-save_4/MergeV2Checkpoints/checkpoint_prefixessave_4/Const*
delete_old_dirs(

save_4/IdentityIdentitysave_4/Const^save_4/MergeV2Checkpoints^save_4/control_dependency*
_output_shapes
: *
T0
¹
save_4/RestoreV2/tensor_namesConst*ē
valueŻBŚ.Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/actor/actor/dense/biasBpi/actor/actor/dense/bias/AdamB pi/actor/actor/dense/bias/Adam_1Bpi/actor/actor/dense/kernelB pi/actor/actor/dense/kernel/AdamB"pi/actor/actor/dense/kernel/Adam_1Bpi/actor/actor/dense_1/biasB pi/actor/actor/dense_1/bias/AdamB"pi/actor/actor/dense_1/bias/Adam_1Bpi/actor/actor/dense_1/kernelB"pi/actor/actor/dense_1/kernel/AdamB$pi/actor/actor/dense_1/kernel/Adam_1Bpi/actor/actor/dense_2/biasB pi/actor/actor/dense_2/bias/AdamB"pi/actor/actor/dense_2/bias/Adam_1Bpi/actor/actor/dense_2/kernelB"pi/actor/actor/dense_2/kernel/AdamB$pi/actor/actor/dense_2/kernel/Adam_1Bpi/actor/actor/dense_3/biasB pi/actor/actor/dense_3/bias/AdamB"pi/actor/actor/dense_3/bias/Adam_1Bpi/actor/actor/dense_3/kernelB"pi/actor/actor/dense_3/kernel/AdamB$pi/actor/actor/dense_3/kernel/Adam_1Bv/critic/critic/dense_4/biasB!v/critic/critic/dense_4/bias/AdamB#v/critic/critic/dense_4/bias/Adam_1Bv/critic/critic/dense_4/kernelB#v/critic/critic/dense_4/kernel/AdamB%v/critic/critic/dense_4/kernel/Adam_1Bv/critic/critic/dense_5/biasB!v/critic/critic/dense_5/bias/AdamB#v/critic/critic/dense_5/bias/Adam_1Bv/critic/critic/dense_5/kernelB#v/critic/critic/dense_5/kernel/AdamB%v/critic/critic/dense_5/kernel/Adam_1Bv/critic/critic/dense_6/biasB!v/critic/critic/dense_6/bias/AdamB#v/critic/critic/dense_6/bias/Adam_1Bv/critic/critic/dense_6/kernelB#v/critic/critic/dense_6/kernel/AdamB%v/critic/critic/dense_6/kernel/Adam_1*
dtype0*
_output_shapes
:.
Ä
!save_4/RestoreV2/shape_and_slicesConst*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:.
ü
save_4/RestoreV2	RestoreV2save_4/Constsave_4/RestoreV2/tensor_names!save_4/RestoreV2/shape_and_slices*Ī
_output_shapes»
ø::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.
R
save_4/Identity_1Identitysave_4/RestoreV2*
T0*
_output_shapes
:
X
save_4/AssignVariableOpAssignVariableOpbeta1_powersave_4/Identity_1*
dtype0
T
save_4/Identity_2Identitysave_4/RestoreV2:1*
T0*
_output_shapes
:
\
save_4/AssignVariableOp_1AssignVariableOpbeta1_power_1save_4/Identity_2*
dtype0
T
save_4/Identity_3Identitysave_4/RestoreV2:2*
T0*
_output_shapes
:
Z
save_4/AssignVariableOp_2AssignVariableOpbeta2_powersave_4/Identity_3*
dtype0
T
save_4/Identity_4Identitysave_4/RestoreV2:3*
_output_shapes
:*
T0
\
save_4/AssignVariableOp_3AssignVariableOpbeta2_power_1save_4/Identity_4*
dtype0
T
save_4/Identity_5Identitysave_4/RestoreV2:4*
T0*
_output_shapes
:
h
save_4/AssignVariableOp_4AssignVariableOppi/actor/actor/dense/biassave_4/Identity_5*
dtype0
T
save_4/Identity_6Identitysave_4/RestoreV2:5*
T0*
_output_shapes
:
m
save_4/AssignVariableOp_5AssignVariableOppi/actor/actor/dense/bias/Adamsave_4/Identity_6*
dtype0
T
save_4/Identity_7Identitysave_4/RestoreV2:6*
_output_shapes
:*
T0
o
save_4/AssignVariableOp_6AssignVariableOp pi/actor/actor/dense/bias/Adam_1save_4/Identity_7*
dtype0
T
save_4/Identity_8Identitysave_4/RestoreV2:7*
_output_shapes
:*
T0
j
save_4/AssignVariableOp_7AssignVariableOppi/actor/actor/dense/kernelsave_4/Identity_8*
dtype0
T
save_4/Identity_9Identitysave_4/RestoreV2:8*
T0*
_output_shapes
:
o
save_4/AssignVariableOp_8AssignVariableOp pi/actor/actor/dense/kernel/Adamsave_4/Identity_9*
dtype0
U
save_4/Identity_10Identitysave_4/RestoreV2:9*
_output_shapes
:*
T0
r
save_4/AssignVariableOp_9AssignVariableOp"pi/actor/actor/dense/kernel/Adam_1save_4/Identity_10*
dtype0
V
save_4/Identity_11Identitysave_4/RestoreV2:10*
T0*
_output_shapes
:
l
save_4/AssignVariableOp_10AssignVariableOppi/actor/actor/dense_1/biassave_4/Identity_11*
dtype0
V
save_4/Identity_12Identitysave_4/RestoreV2:11*
T0*
_output_shapes
:
q
save_4/AssignVariableOp_11AssignVariableOp pi/actor/actor/dense_1/bias/Adamsave_4/Identity_12*
dtype0
V
save_4/Identity_13Identitysave_4/RestoreV2:12*
T0*
_output_shapes
:
s
save_4/AssignVariableOp_12AssignVariableOp"pi/actor/actor/dense_1/bias/Adam_1save_4/Identity_13*
dtype0
V
save_4/Identity_14Identitysave_4/RestoreV2:13*
T0*
_output_shapes
:
n
save_4/AssignVariableOp_13AssignVariableOppi/actor/actor/dense_1/kernelsave_4/Identity_14*
dtype0
V
save_4/Identity_15Identitysave_4/RestoreV2:14*
T0*
_output_shapes
:
s
save_4/AssignVariableOp_14AssignVariableOp"pi/actor/actor/dense_1/kernel/Adamsave_4/Identity_15*
dtype0
V
save_4/Identity_16Identitysave_4/RestoreV2:15*
T0*
_output_shapes
:
u
save_4/AssignVariableOp_15AssignVariableOp$pi/actor/actor/dense_1/kernel/Adam_1save_4/Identity_16*
dtype0
V
save_4/Identity_17Identitysave_4/RestoreV2:16*
T0*
_output_shapes
:
l
save_4/AssignVariableOp_16AssignVariableOppi/actor/actor/dense_2/biassave_4/Identity_17*
dtype0
V
save_4/Identity_18Identitysave_4/RestoreV2:17*
T0*
_output_shapes
:
q
save_4/AssignVariableOp_17AssignVariableOp pi/actor/actor/dense_2/bias/Adamsave_4/Identity_18*
dtype0
V
save_4/Identity_19Identitysave_4/RestoreV2:18*
T0*
_output_shapes
:
s
save_4/AssignVariableOp_18AssignVariableOp"pi/actor/actor/dense_2/bias/Adam_1save_4/Identity_19*
dtype0
V
save_4/Identity_20Identitysave_4/RestoreV2:19*
T0*
_output_shapes
:
n
save_4/AssignVariableOp_19AssignVariableOppi/actor/actor/dense_2/kernelsave_4/Identity_20*
dtype0
V
save_4/Identity_21Identitysave_4/RestoreV2:20*
T0*
_output_shapes
:
s
save_4/AssignVariableOp_20AssignVariableOp"pi/actor/actor/dense_2/kernel/Adamsave_4/Identity_21*
dtype0
V
save_4/Identity_22Identitysave_4/RestoreV2:21*
T0*
_output_shapes
:
u
save_4/AssignVariableOp_21AssignVariableOp$pi/actor/actor/dense_2/kernel/Adam_1save_4/Identity_22*
dtype0
V
save_4/Identity_23Identitysave_4/RestoreV2:22*
T0*
_output_shapes
:
l
save_4/AssignVariableOp_22AssignVariableOppi/actor/actor/dense_3/biassave_4/Identity_23*
dtype0
V
save_4/Identity_24Identitysave_4/RestoreV2:23*
_output_shapes
:*
T0
q
save_4/AssignVariableOp_23AssignVariableOp pi/actor/actor/dense_3/bias/Adamsave_4/Identity_24*
dtype0
V
save_4/Identity_25Identitysave_4/RestoreV2:24*
T0*
_output_shapes
:
s
save_4/AssignVariableOp_24AssignVariableOp"pi/actor/actor/dense_3/bias/Adam_1save_4/Identity_25*
dtype0
V
save_4/Identity_26Identitysave_4/RestoreV2:25*
_output_shapes
:*
T0
n
save_4/AssignVariableOp_25AssignVariableOppi/actor/actor/dense_3/kernelsave_4/Identity_26*
dtype0
V
save_4/Identity_27Identitysave_4/RestoreV2:26*
T0*
_output_shapes
:
s
save_4/AssignVariableOp_26AssignVariableOp"pi/actor/actor/dense_3/kernel/Adamsave_4/Identity_27*
dtype0
V
save_4/Identity_28Identitysave_4/RestoreV2:27*
_output_shapes
:*
T0
u
save_4/AssignVariableOp_27AssignVariableOp$pi/actor/actor/dense_3/kernel/Adam_1save_4/Identity_28*
dtype0
V
save_4/Identity_29Identitysave_4/RestoreV2:28*
_output_shapes
:*
T0
m
save_4/AssignVariableOp_28AssignVariableOpv/critic/critic/dense_4/biassave_4/Identity_29*
dtype0
V
save_4/Identity_30Identitysave_4/RestoreV2:29*
_output_shapes
:*
T0
r
save_4/AssignVariableOp_29AssignVariableOp!v/critic/critic/dense_4/bias/Adamsave_4/Identity_30*
dtype0
V
save_4/Identity_31Identitysave_4/RestoreV2:30*
T0*
_output_shapes
:
t
save_4/AssignVariableOp_30AssignVariableOp#v/critic/critic/dense_4/bias/Adam_1save_4/Identity_31*
dtype0
V
save_4/Identity_32Identitysave_4/RestoreV2:31*
T0*
_output_shapes
:
o
save_4/AssignVariableOp_31AssignVariableOpv/critic/critic/dense_4/kernelsave_4/Identity_32*
dtype0
V
save_4/Identity_33Identitysave_4/RestoreV2:32*
T0*
_output_shapes
:
t
save_4/AssignVariableOp_32AssignVariableOp#v/critic/critic/dense_4/kernel/Adamsave_4/Identity_33*
dtype0
V
save_4/Identity_34Identitysave_4/RestoreV2:33*
T0*
_output_shapes
:
v
save_4/AssignVariableOp_33AssignVariableOp%v/critic/critic/dense_4/kernel/Adam_1save_4/Identity_34*
dtype0
V
save_4/Identity_35Identitysave_4/RestoreV2:34*
_output_shapes
:*
T0
m
save_4/AssignVariableOp_34AssignVariableOpv/critic/critic/dense_5/biassave_4/Identity_35*
dtype0
V
save_4/Identity_36Identitysave_4/RestoreV2:35*
T0*
_output_shapes
:
r
save_4/AssignVariableOp_35AssignVariableOp!v/critic/critic/dense_5/bias/Adamsave_4/Identity_36*
dtype0
V
save_4/Identity_37Identitysave_4/RestoreV2:36*
_output_shapes
:*
T0
t
save_4/AssignVariableOp_36AssignVariableOp#v/critic/critic/dense_5/bias/Adam_1save_4/Identity_37*
dtype0
V
save_4/Identity_38Identitysave_4/RestoreV2:37*
_output_shapes
:*
T0
o
save_4/AssignVariableOp_37AssignVariableOpv/critic/critic/dense_5/kernelsave_4/Identity_38*
dtype0
V
save_4/Identity_39Identitysave_4/RestoreV2:38*
T0*
_output_shapes
:
t
save_4/AssignVariableOp_38AssignVariableOp#v/critic/critic/dense_5/kernel/Adamsave_4/Identity_39*
dtype0
V
save_4/Identity_40Identitysave_4/RestoreV2:39*
_output_shapes
:*
T0
v
save_4/AssignVariableOp_39AssignVariableOp%v/critic/critic/dense_5/kernel/Adam_1save_4/Identity_40*
dtype0
V
save_4/Identity_41Identitysave_4/RestoreV2:40*
_output_shapes
:*
T0
m
save_4/AssignVariableOp_40AssignVariableOpv/critic/critic/dense_6/biassave_4/Identity_41*
dtype0
V
save_4/Identity_42Identitysave_4/RestoreV2:41*
T0*
_output_shapes
:
r
save_4/AssignVariableOp_41AssignVariableOp!v/critic/critic/dense_6/bias/Adamsave_4/Identity_42*
dtype0
V
save_4/Identity_43Identitysave_4/RestoreV2:42*
T0*
_output_shapes
:
t
save_4/AssignVariableOp_42AssignVariableOp#v/critic/critic/dense_6/bias/Adam_1save_4/Identity_43*
dtype0
V
save_4/Identity_44Identitysave_4/RestoreV2:43*
T0*
_output_shapes
:
o
save_4/AssignVariableOp_43AssignVariableOpv/critic/critic/dense_6/kernelsave_4/Identity_44*
dtype0
V
save_4/Identity_45Identitysave_4/RestoreV2:44*
_output_shapes
:*
T0
t
save_4/AssignVariableOp_44AssignVariableOp#v/critic/critic/dense_6/kernel/Adamsave_4/Identity_45*
dtype0
V
save_4/Identity_46Identitysave_4/RestoreV2:45*
T0*
_output_shapes
:
v
save_4/AssignVariableOp_45AssignVariableOp%v/critic/critic/dense_6/kernel/Adam_1save_4/Identity_46*
dtype0
Ę

save_4/restore_shardNoOp^save_4/AssignVariableOp^save_4/AssignVariableOp_1^save_4/AssignVariableOp_10^save_4/AssignVariableOp_11^save_4/AssignVariableOp_12^save_4/AssignVariableOp_13^save_4/AssignVariableOp_14^save_4/AssignVariableOp_15^save_4/AssignVariableOp_16^save_4/AssignVariableOp_17^save_4/AssignVariableOp_18^save_4/AssignVariableOp_19^save_4/AssignVariableOp_2^save_4/AssignVariableOp_20^save_4/AssignVariableOp_21^save_4/AssignVariableOp_22^save_4/AssignVariableOp_23^save_4/AssignVariableOp_24^save_4/AssignVariableOp_25^save_4/AssignVariableOp_26^save_4/AssignVariableOp_27^save_4/AssignVariableOp_28^save_4/AssignVariableOp_29^save_4/AssignVariableOp_3^save_4/AssignVariableOp_30^save_4/AssignVariableOp_31^save_4/AssignVariableOp_32^save_4/AssignVariableOp_33^save_4/AssignVariableOp_34^save_4/AssignVariableOp_35^save_4/AssignVariableOp_36^save_4/AssignVariableOp_37^save_4/AssignVariableOp_38^save_4/AssignVariableOp_39^save_4/AssignVariableOp_4^save_4/AssignVariableOp_40^save_4/AssignVariableOp_41^save_4/AssignVariableOp_42^save_4/AssignVariableOp_43^save_4/AssignVariableOp_44^save_4/AssignVariableOp_45^save_4/AssignVariableOp_5^save_4/AssignVariableOp_6^save_4/AssignVariableOp_7^save_4/AssignVariableOp_8^save_4/AssignVariableOp_9
1
save_4/restore_allNoOp^save_4/restore_shard
[
save_5/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_5/filenamePlaceholderWithDefaultsave_5/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_5/ConstPlaceholderWithDefaultsave_5/filename*
_output_shapes
: *
shape: *
dtype0

save_5/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_f7e685c97daf4d63ba161f4d6e65a3f7/part*
dtype0
{
save_5/StringJoin
StringJoinsave_5/Constsave_5/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_5/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_5/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_5/ShardedFilenameShardedFilenamesave_5/StringJoinsave_5/ShardedFilename/shardsave_5/num_shards*
_output_shapes
: 
¶
save_5/SaveV2/tensor_namesConst*ē
valueŻBŚ.Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/actor/actor/dense/biasBpi/actor/actor/dense/bias/AdamB pi/actor/actor/dense/bias/Adam_1Bpi/actor/actor/dense/kernelB pi/actor/actor/dense/kernel/AdamB"pi/actor/actor/dense/kernel/Adam_1Bpi/actor/actor/dense_1/biasB pi/actor/actor/dense_1/bias/AdamB"pi/actor/actor/dense_1/bias/Adam_1Bpi/actor/actor/dense_1/kernelB"pi/actor/actor/dense_1/kernel/AdamB$pi/actor/actor/dense_1/kernel/Adam_1Bpi/actor/actor/dense_2/biasB pi/actor/actor/dense_2/bias/AdamB"pi/actor/actor/dense_2/bias/Adam_1Bpi/actor/actor/dense_2/kernelB"pi/actor/actor/dense_2/kernel/AdamB$pi/actor/actor/dense_2/kernel/Adam_1Bpi/actor/actor/dense_3/biasB pi/actor/actor/dense_3/bias/AdamB"pi/actor/actor/dense_3/bias/Adam_1Bpi/actor/actor/dense_3/kernelB"pi/actor/actor/dense_3/kernel/AdamB$pi/actor/actor/dense_3/kernel/Adam_1Bv/critic/critic/dense_4/biasB!v/critic/critic/dense_4/bias/AdamB#v/critic/critic/dense_4/bias/Adam_1Bv/critic/critic/dense_4/kernelB#v/critic/critic/dense_4/kernel/AdamB%v/critic/critic/dense_4/kernel/Adam_1Bv/critic/critic/dense_5/biasB!v/critic/critic/dense_5/bias/AdamB#v/critic/critic/dense_5/bias/Adam_1Bv/critic/critic/dense_5/kernelB#v/critic/critic/dense_5/kernel/AdamB%v/critic/critic/dense_5/kernel/Adam_1Bv/critic/critic/dense_6/biasB!v/critic/critic/dense_6/bias/AdamB#v/critic/critic/dense_6/bias/Adam_1Bv/critic/critic/dense_6/kernelB#v/critic/critic/dense_6/kernel/AdamB%v/critic/critic/dense_6/kernel/Adam_1*
dtype0*
_output_shapes
:.
Į
save_5/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:.*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 

save_5/SaveV2SaveV2save_5/ShardedFilenamesave_5/SaveV2/tensor_namessave_5/SaveV2/shape_and_slicesbeta1_power/Read/ReadVariableOp!beta1_power_1/Read/ReadVariableOpbeta2_power/Read/ReadVariableOp!beta2_power_1/Read/ReadVariableOp-pi/actor/actor/dense/bias/Read/ReadVariableOp2pi/actor/actor/dense/bias/Adam/Read/ReadVariableOp4pi/actor/actor/dense/bias/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense/kernel/Read/ReadVariableOp4pi/actor/actor/dense/kernel/Adam/Read/ReadVariableOp6pi/actor/actor/dense/kernel/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense_1/bias/Read/ReadVariableOp4pi/actor/actor/dense_1/bias/Adam/Read/ReadVariableOp6pi/actor/actor/dense_1/bias/Adam_1/Read/ReadVariableOp1pi/actor/actor/dense_1/kernel/Read/ReadVariableOp6pi/actor/actor/dense_1/kernel/Adam/Read/ReadVariableOp8pi/actor/actor/dense_1/kernel/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense_2/bias/Read/ReadVariableOp4pi/actor/actor/dense_2/bias/Adam/Read/ReadVariableOp6pi/actor/actor/dense_2/bias/Adam_1/Read/ReadVariableOp1pi/actor/actor/dense_2/kernel/Read/ReadVariableOp6pi/actor/actor/dense_2/kernel/Adam/Read/ReadVariableOp8pi/actor/actor/dense_2/kernel/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense_3/bias/Read/ReadVariableOp4pi/actor/actor/dense_3/bias/Adam/Read/ReadVariableOp6pi/actor/actor/dense_3/bias/Adam_1/Read/ReadVariableOp1pi/actor/actor/dense_3/kernel/Read/ReadVariableOp6pi/actor/actor/dense_3/kernel/Adam/Read/ReadVariableOp8pi/actor/actor/dense_3/kernel/Adam_1/Read/ReadVariableOp0v/critic/critic/dense_4/bias/Read/ReadVariableOp5v/critic/critic/dense_4/bias/Adam/Read/ReadVariableOp7v/critic/critic/dense_4/bias/Adam_1/Read/ReadVariableOp2v/critic/critic/dense_4/kernel/Read/ReadVariableOp7v/critic/critic/dense_4/kernel/Adam/Read/ReadVariableOp9v/critic/critic/dense_4/kernel/Adam_1/Read/ReadVariableOp0v/critic/critic/dense_5/bias/Read/ReadVariableOp5v/critic/critic/dense_5/bias/Adam/Read/ReadVariableOp7v/critic/critic/dense_5/bias/Adam_1/Read/ReadVariableOp2v/critic/critic/dense_5/kernel/Read/ReadVariableOp7v/critic/critic/dense_5/kernel/Adam/Read/ReadVariableOp9v/critic/critic/dense_5/kernel/Adam_1/Read/ReadVariableOp0v/critic/critic/dense_6/bias/Read/ReadVariableOp5v/critic/critic/dense_6/bias/Adam/Read/ReadVariableOp7v/critic/critic/dense_6/bias/Adam_1/Read/ReadVariableOp2v/critic/critic/dense_6/kernel/Read/ReadVariableOp7v/critic/critic/dense_6/kernel/Adam/Read/ReadVariableOp9v/critic/critic/dense_6/kernel/Adam_1/Read/ReadVariableOp*<
dtypes2
02.

save_5/control_dependencyIdentitysave_5/ShardedFilename^save_5/SaveV2*
T0*)
_class
loc:@save_5/ShardedFilename*
_output_shapes
: 
£
-save_5/MergeV2Checkpoints/checkpoint_prefixesPacksave_5/ShardedFilename^save_5/control_dependency*
N*
_output_shapes
:*
T0*

axis 

save_5/MergeV2CheckpointsMergeV2Checkpoints-save_5/MergeV2Checkpoints/checkpoint_prefixessave_5/Const*
delete_old_dirs(

save_5/IdentityIdentitysave_5/Const^save_5/MergeV2Checkpoints^save_5/control_dependency*
T0*
_output_shapes
: 
¹
save_5/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:.*ē
valueŻBŚ.Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/actor/actor/dense/biasBpi/actor/actor/dense/bias/AdamB pi/actor/actor/dense/bias/Adam_1Bpi/actor/actor/dense/kernelB pi/actor/actor/dense/kernel/AdamB"pi/actor/actor/dense/kernel/Adam_1Bpi/actor/actor/dense_1/biasB pi/actor/actor/dense_1/bias/AdamB"pi/actor/actor/dense_1/bias/Adam_1Bpi/actor/actor/dense_1/kernelB"pi/actor/actor/dense_1/kernel/AdamB$pi/actor/actor/dense_1/kernel/Adam_1Bpi/actor/actor/dense_2/biasB pi/actor/actor/dense_2/bias/AdamB"pi/actor/actor/dense_2/bias/Adam_1Bpi/actor/actor/dense_2/kernelB"pi/actor/actor/dense_2/kernel/AdamB$pi/actor/actor/dense_2/kernel/Adam_1Bpi/actor/actor/dense_3/biasB pi/actor/actor/dense_3/bias/AdamB"pi/actor/actor/dense_3/bias/Adam_1Bpi/actor/actor/dense_3/kernelB"pi/actor/actor/dense_3/kernel/AdamB$pi/actor/actor/dense_3/kernel/Adam_1Bv/critic/critic/dense_4/biasB!v/critic/critic/dense_4/bias/AdamB#v/critic/critic/dense_4/bias/Adam_1Bv/critic/critic/dense_4/kernelB#v/critic/critic/dense_4/kernel/AdamB%v/critic/critic/dense_4/kernel/Adam_1Bv/critic/critic/dense_5/biasB!v/critic/critic/dense_5/bias/AdamB#v/critic/critic/dense_5/bias/Adam_1Bv/critic/critic/dense_5/kernelB#v/critic/critic/dense_5/kernel/AdamB%v/critic/critic/dense_5/kernel/Adam_1Bv/critic/critic/dense_6/biasB!v/critic/critic/dense_6/bias/AdamB#v/critic/critic/dense_6/bias/Adam_1Bv/critic/critic/dense_6/kernelB#v/critic/critic/dense_6/kernel/AdamB%v/critic/critic/dense_6/kernel/Adam_1
Ä
!save_5/RestoreV2/shape_and_slicesConst*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:.
ü
save_5/RestoreV2	RestoreV2save_5/Constsave_5/RestoreV2/tensor_names!save_5/RestoreV2/shape_and_slices*Ī
_output_shapes»
ø::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.
R
save_5/Identity_1Identitysave_5/RestoreV2*
T0*
_output_shapes
:
X
save_5/AssignVariableOpAssignVariableOpbeta1_powersave_5/Identity_1*
dtype0
T
save_5/Identity_2Identitysave_5/RestoreV2:1*
_output_shapes
:*
T0
\
save_5/AssignVariableOp_1AssignVariableOpbeta1_power_1save_5/Identity_2*
dtype0
T
save_5/Identity_3Identitysave_5/RestoreV2:2*
_output_shapes
:*
T0
Z
save_5/AssignVariableOp_2AssignVariableOpbeta2_powersave_5/Identity_3*
dtype0
T
save_5/Identity_4Identitysave_5/RestoreV2:3*
T0*
_output_shapes
:
\
save_5/AssignVariableOp_3AssignVariableOpbeta2_power_1save_5/Identity_4*
dtype0
T
save_5/Identity_5Identitysave_5/RestoreV2:4*
T0*
_output_shapes
:
h
save_5/AssignVariableOp_4AssignVariableOppi/actor/actor/dense/biassave_5/Identity_5*
dtype0
T
save_5/Identity_6Identitysave_5/RestoreV2:5*
_output_shapes
:*
T0
m
save_5/AssignVariableOp_5AssignVariableOppi/actor/actor/dense/bias/Adamsave_5/Identity_6*
dtype0
T
save_5/Identity_7Identitysave_5/RestoreV2:6*
_output_shapes
:*
T0
o
save_5/AssignVariableOp_6AssignVariableOp pi/actor/actor/dense/bias/Adam_1save_5/Identity_7*
dtype0
T
save_5/Identity_8Identitysave_5/RestoreV2:7*
_output_shapes
:*
T0
j
save_5/AssignVariableOp_7AssignVariableOppi/actor/actor/dense/kernelsave_5/Identity_8*
dtype0
T
save_5/Identity_9Identitysave_5/RestoreV2:8*
T0*
_output_shapes
:
o
save_5/AssignVariableOp_8AssignVariableOp pi/actor/actor/dense/kernel/Adamsave_5/Identity_9*
dtype0
U
save_5/Identity_10Identitysave_5/RestoreV2:9*
T0*
_output_shapes
:
r
save_5/AssignVariableOp_9AssignVariableOp"pi/actor/actor/dense/kernel/Adam_1save_5/Identity_10*
dtype0
V
save_5/Identity_11Identitysave_5/RestoreV2:10*
_output_shapes
:*
T0
l
save_5/AssignVariableOp_10AssignVariableOppi/actor/actor/dense_1/biassave_5/Identity_11*
dtype0
V
save_5/Identity_12Identitysave_5/RestoreV2:11*
T0*
_output_shapes
:
q
save_5/AssignVariableOp_11AssignVariableOp pi/actor/actor/dense_1/bias/Adamsave_5/Identity_12*
dtype0
V
save_5/Identity_13Identitysave_5/RestoreV2:12*
T0*
_output_shapes
:
s
save_5/AssignVariableOp_12AssignVariableOp"pi/actor/actor/dense_1/bias/Adam_1save_5/Identity_13*
dtype0
V
save_5/Identity_14Identitysave_5/RestoreV2:13*
T0*
_output_shapes
:
n
save_5/AssignVariableOp_13AssignVariableOppi/actor/actor/dense_1/kernelsave_5/Identity_14*
dtype0
V
save_5/Identity_15Identitysave_5/RestoreV2:14*
T0*
_output_shapes
:
s
save_5/AssignVariableOp_14AssignVariableOp"pi/actor/actor/dense_1/kernel/Adamsave_5/Identity_15*
dtype0
V
save_5/Identity_16Identitysave_5/RestoreV2:15*
_output_shapes
:*
T0
u
save_5/AssignVariableOp_15AssignVariableOp$pi/actor/actor/dense_1/kernel/Adam_1save_5/Identity_16*
dtype0
V
save_5/Identity_17Identitysave_5/RestoreV2:16*
_output_shapes
:*
T0
l
save_5/AssignVariableOp_16AssignVariableOppi/actor/actor/dense_2/biassave_5/Identity_17*
dtype0
V
save_5/Identity_18Identitysave_5/RestoreV2:17*
_output_shapes
:*
T0
q
save_5/AssignVariableOp_17AssignVariableOp pi/actor/actor/dense_2/bias/Adamsave_5/Identity_18*
dtype0
V
save_5/Identity_19Identitysave_5/RestoreV2:18*
T0*
_output_shapes
:
s
save_5/AssignVariableOp_18AssignVariableOp"pi/actor/actor/dense_2/bias/Adam_1save_5/Identity_19*
dtype0
V
save_5/Identity_20Identitysave_5/RestoreV2:19*
_output_shapes
:*
T0
n
save_5/AssignVariableOp_19AssignVariableOppi/actor/actor/dense_2/kernelsave_5/Identity_20*
dtype0
V
save_5/Identity_21Identitysave_5/RestoreV2:20*
T0*
_output_shapes
:
s
save_5/AssignVariableOp_20AssignVariableOp"pi/actor/actor/dense_2/kernel/Adamsave_5/Identity_21*
dtype0
V
save_5/Identity_22Identitysave_5/RestoreV2:21*
T0*
_output_shapes
:
u
save_5/AssignVariableOp_21AssignVariableOp$pi/actor/actor/dense_2/kernel/Adam_1save_5/Identity_22*
dtype0
V
save_5/Identity_23Identitysave_5/RestoreV2:22*
_output_shapes
:*
T0
l
save_5/AssignVariableOp_22AssignVariableOppi/actor/actor/dense_3/biassave_5/Identity_23*
dtype0
V
save_5/Identity_24Identitysave_5/RestoreV2:23*
_output_shapes
:*
T0
q
save_5/AssignVariableOp_23AssignVariableOp pi/actor/actor/dense_3/bias/Adamsave_5/Identity_24*
dtype0
V
save_5/Identity_25Identitysave_5/RestoreV2:24*
_output_shapes
:*
T0
s
save_5/AssignVariableOp_24AssignVariableOp"pi/actor/actor/dense_3/bias/Adam_1save_5/Identity_25*
dtype0
V
save_5/Identity_26Identitysave_5/RestoreV2:25*
T0*
_output_shapes
:
n
save_5/AssignVariableOp_25AssignVariableOppi/actor/actor/dense_3/kernelsave_5/Identity_26*
dtype0
V
save_5/Identity_27Identitysave_5/RestoreV2:26*
_output_shapes
:*
T0
s
save_5/AssignVariableOp_26AssignVariableOp"pi/actor/actor/dense_3/kernel/Adamsave_5/Identity_27*
dtype0
V
save_5/Identity_28Identitysave_5/RestoreV2:27*
_output_shapes
:*
T0
u
save_5/AssignVariableOp_27AssignVariableOp$pi/actor/actor/dense_3/kernel/Adam_1save_5/Identity_28*
dtype0
V
save_5/Identity_29Identitysave_5/RestoreV2:28*
T0*
_output_shapes
:
m
save_5/AssignVariableOp_28AssignVariableOpv/critic/critic/dense_4/biassave_5/Identity_29*
dtype0
V
save_5/Identity_30Identitysave_5/RestoreV2:29*
_output_shapes
:*
T0
r
save_5/AssignVariableOp_29AssignVariableOp!v/critic/critic/dense_4/bias/Adamsave_5/Identity_30*
dtype0
V
save_5/Identity_31Identitysave_5/RestoreV2:30*
T0*
_output_shapes
:
t
save_5/AssignVariableOp_30AssignVariableOp#v/critic/critic/dense_4/bias/Adam_1save_5/Identity_31*
dtype0
V
save_5/Identity_32Identitysave_5/RestoreV2:31*
_output_shapes
:*
T0
o
save_5/AssignVariableOp_31AssignVariableOpv/critic/critic/dense_4/kernelsave_5/Identity_32*
dtype0
V
save_5/Identity_33Identitysave_5/RestoreV2:32*
T0*
_output_shapes
:
t
save_5/AssignVariableOp_32AssignVariableOp#v/critic/critic/dense_4/kernel/Adamsave_5/Identity_33*
dtype0
V
save_5/Identity_34Identitysave_5/RestoreV2:33*
_output_shapes
:*
T0
v
save_5/AssignVariableOp_33AssignVariableOp%v/critic/critic/dense_4/kernel/Adam_1save_5/Identity_34*
dtype0
V
save_5/Identity_35Identitysave_5/RestoreV2:34*
_output_shapes
:*
T0
m
save_5/AssignVariableOp_34AssignVariableOpv/critic/critic/dense_5/biassave_5/Identity_35*
dtype0
V
save_5/Identity_36Identitysave_5/RestoreV2:35*
T0*
_output_shapes
:
r
save_5/AssignVariableOp_35AssignVariableOp!v/critic/critic/dense_5/bias/Adamsave_5/Identity_36*
dtype0
V
save_5/Identity_37Identitysave_5/RestoreV2:36*
T0*
_output_shapes
:
t
save_5/AssignVariableOp_36AssignVariableOp#v/critic/critic/dense_5/bias/Adam_1save_5/Identity_37*
dtype0
V
save_5/Identity_38Identitysave_5/RestoreV2:37*
_output_shapes
:*
T0
o
save_5/AssignVariableOp_37AssignVariableOpv/critic/critic/dense_5/kernelsave_5/Identity_38*
dtype0
V
save_5/Identity_39Identitysave_5/RestoreV2:38*
_output_shapes
:*
T0
t
save_5/AssignVariableOp_38AssignVariableOp#v/critic/critic/dense_5/kernel/Adamsave_5/Identity_39*
dtype0
V
save_5/Identity_40Identitysave_5/RestoreV2:39*
T0*
_output_shapes
:
v
save_5/AssignVariableOp_39AssignVariableOp%v/critic/critic/dense_5/kernel/Adam_1save_5/Identity_40*
dtype0
V
save_5/Identity_41Identitysave_5/RestoreV2:40*
_output_shapes
:*
T0
m
save_5/AssignVariableOp_40AssignVariableOpv/critic/critic/dense_6/biassave_5/Identity_41*
dtype0
V
save_5/Identity_42Identitysave_5/RestoreV2:41*
_output_shapes
:*
T0
r
save_5/AssignVariableOp_41AssignVariableOp!v/critic/critic/dense_6/bias/Adamsave_5/Identity_42*
dtype0
V
save_5/Identity_43Identitysave_5/RestoreV2:42*
_output_shapes
:*
T0
t
save_5/AssignVariableOp_42AssignVariableOp#v/critic/critic/dense_6/bias/Adam_1save_5/Identity_43*
dtype0
V
save_5/Identity_44Identitysave_5/RestoreV2:43*
_output_shapes
:*
T0
o
save_5/AssignVariableOp_43AssignVariableOpv/critic/critic/dense_6/kernelsave_5/Identity_44*
dtype0
V
save_5/Identity_45Identitysave_5/RestoreV2:44*
T0*
_output_shapes
:
t
save_5/AssignVariableOp_44AssignVariableOp#v/critic/critic/dense_6/kernel/Adamsave_5/Identity_45*
dtype0
V
save_5/Identity_46Identitysave_5/RestoreV2:45*
T0*
_output_shapes
:
v
save_5/AssignVariableOp_45AssignVariableOp%v/critic/critic/dense_6/kernel/Adam_1save_5/Identity_46*
dtype0
Ę

save_5/restore_shardNoOp^save_5/AssignVariableOp^save_5/AssignVariableOp_1^save_5/AssignVariableOp_10^save_5/AssignVariableOp_11^save_5/AssignVariableOp_12^save_5/AssignVariableOp_13^save_5/AssignVariableOp_14^save_5/AssignVariableOp_15^save_5/AssignVariableOp_16^save_5/AssignVariableOp_17^save_5/AssignVariableOp_18^save_5/AssignVariableOp_19^save_5/AssignVariableOp_2^save_5/AssignVariableOp_20^save_5/AssignVariableOp_21^save_5/AssignVariableOp_22^save_5/AssignVariableOp_23^save_5/AssignVariableOp_24^save_5/AssignVariableOp_25^save_5/AssignVariableOp_26^save_5/AssignVariableOp_27^save_5/AssignVariableOp_28^save_5/AssignVariableOp_29^save_5/AssignVariableOp_3^save_5/AssignVariableOp_30^save_5/AssignVariableOp_31^save_5/AssignVariableOp_32^save_5/AssignVariableOp_33^save_5/AssignVariableOp_34^save_5/AssignVariableOp_35^save_5/AssignVariableOp_36^save_5/AssignVariableOp_37^save_5/AssignVariableOp_38^save_5/AssignVariableOp_39^save_5/AssignVariableOp_4^save_5/AssignVariableOp_40^save_5/AssignVariableOp_41^save_5/AssignVariableOp_42^save_5/AssignVariableOp_43^save_5/AssignVariableOp_44^save_5/AssignVariableOp_45^save_5/AssignVariableOp_5^save_5/AssignVariableOp_6^save_5/AssignVariableOp_7^save_5/AssignVariableOp_8^save_5/AssignVariableOp_9
1
save_5/restore_allNoOp^save_5/restore_shard
[
save_6/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_6/filenamePlaceholderWithDefaultsave_6/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_6/ConstPlaceholderWithDefaultsave_6/filename*
dtype0*
_output_shapes
: *
shape: 

save_6/StringJoin/inputs_1Const*<
value3B1 B+_temp_914e93a696db4f84a56fa6f4efd40090/part*
dtype0*
_output_shapes
: 
{
save_6/StringJoin
StringJoinsave_6/Constsave_6/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_6/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_6/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_6/ShardedFilenameShardedFilenamesave_6/StringJoinsave_6/ShardedFilename/shardsave_6/num_shards*
_output_shapes
: 
¶
save_6/SaveV2/tensor_namesConst*ē
valueŻBŚ.Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/actor/actor/dense/biasBpi/actor/actor/dense/bias/AdamB pi/actor/actor/dense/bias/Adam_1Bpi/actor/actor/dense/kernelB pi/actor/actor/dense/kernel/AdamB"pi/actor/actor/dense/kernel/Adam_1Bpi/actor/actor/dense_1/biasB pi/actor/actor/dense_1/bias/AdamB"pi/actor/actor/dense_1/bias/Adam_1Bpi/actor/actor/dense_1/kernelB"pi/actor/actor/dense_1/kernel/AdamB$pi/actor/actor/dense_1/kernel/Adam_1Bpi/actor/actor/dense_2/biasB pi/actor/actor/dense_2/bias/AdamB"pi/actor/actor/dense_2/bias/Adam_1Bpi/actor/actor/dense_2/kernelB"pi/actor/actor/dense_2/kernel/AdamB$pi/actor/actor/dense_2/kernel/Adam_1Bpi/actor/actor/dense_3/biasB pi/actor/actor/dense_3/bias/AdamB"pi/actor/actor/dense_3/bias/Adam_1Bpi/actor/actor/dense_3/kernelB"pi/actor/actor/dense_3/kernel/AdamB$pi/actor/actor/dense_3/kernel/Adam_1Bv/critic/critic/dense_4/biasB!v/critic/critic/dense_4/bias/AdamB#v/critic/critic/dense_4/bias/Adam_1Bv/critic/critic/dense_4/kernelB#v/critic/critic/dense_4/kernel/AdamB%v/critic/critic/dense_4/kernel/Adam_1Bv/critic/critic/dense_5/biasB!v/critic/critic/dense_5/bias/AdamB#v/critic/critic/dense_5/bias/Adam_1Bv/critic/critic/dense_5/kernelB#v/critic/critic/dense_5/kernel/AdamB%v/critic/critic/dense_5/kernel/Adam_1Bv/critic/critic/dense_6/biasB!v/critic/critic/dense_6/bias/AdamB#v/critic/critic/dense_6/bias/Adam_1Bv/critic/critic/dense_6/kernelB#v/critic/critic/dense_6/kernel/AdamB%v/critic/critic/dense_6/kernel/Adam_1*
dtype0*
_output_shapes
:.
Į
save_6/SaveV2/shape_and_slicesConst*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:.

save_6/SaveV2SaveV2save_6/ShardedFilenamesave_6/SaveV2/tensor_namessave_6/SaveV2/shape_and_slicesbeta1_power/Read/ReadVariableOp!beta1_power_1/Read/ReadVariableOpbeta2_power/Read/ReadVariableOp!beta2_power_1/Read/ReadVariableOp-pi/actor/actor/dense/bias/Read/ReadVariableOp2pi/actor/actor/dense/bias/Adam/Read/ReadVariableOp4pi/actor/actor/dense/bias/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense/kernel/Read/ReadVariableOp4pi/actor/actor/dense/kernel/Adam/Read/ReadVariableOp6pi/actor/actor/dense/kernel/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense_1/bias/Read/ReadVariableOp4pi/actor/actor/dense_1/bias/Adam/Read/ReadVariableOp6pi/actor/actor/dense_1/bias/Adam_1/Read/ReadVariableOp1pi/actor/actor/dense_1/kernel/Read/ReadVariableOp6pi/actor/actor/dense_1/kernel/Adam/Read/ReadVariableOp8pi/actor/actor/dense_1/kernel/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense_2/bias/Read/ReadVariableOp4pi/actor/actor/dense_2/bias/Adam/Read/ReadVariableOp6pi/actor/actor/dense_2/bias/Adam_1/Read/ReadVariableOp1pi/actor/actor/dense_2/kernel/Read/ReadVariableOp6pi/actor/actor/dense_2/kernel/Adam/Read/ReadVariableOp8pi/actor/actor/dense_2/kernel/Adam_1/Read/ReadVariableOp/pi/actor/actor/dense_3/bias/Read/ReadVariableOp4pi/actor/actor/dense_3/bias/Adam/Read/ReadVariableOp6pi/actor/actor/dense_3/bias/Adam_1/Read/ReadVariableOp1pi/actor/actor/dense_3/kernel/Read/ReadVariableOp6pi/actor/actor/dense_3/kernel/Adam/Read/ReadVariableOp8pi/actor/actor/dense_3/kernel/Adam_1/Read/ReadVariableOp0v/critic/critic/dense_4/bias/Read/ReadVariableOp5v/critic/critic/dense_4/bias/Adam/Read/ReadVariableOp7v/critic/critic/dense_4/bias/Adam_1/Read/ReadVariableOp2v/critic/critic/dense_4/kernel/Read/ReadVariableOp7v/critic/critic/dense_4/kernel/Adam/Read/ReadVariableOp9v/critic/critic/dense_4/kernel/Adam_1/Read/ReadVariableOp0v/critic/critic/dense_5/bias/Read/ReadVariableOp5v/critic/critic/dense_5/bias/Adam/Read/ReadVariableOp7v/critic/critic/dense_5/bias/Adam_1/Read/ReadVariableOp2v/critic/critic/dense_5/kernel/Read/ReadVariableOp7v/critic/critic/dense_5/kernel/Adam/Read/ReadVariableOp9v/critic/critic/dense_5/kernel/Adam_1/Read/ReadVariableOp0v/critic/critic/dense_6/bias/Read/ReadVariableOp5v/critic/critic/dense_6/bias/Adam/Read/ReadVariableOp7v/critic/critic/dense_6/bias/Adam_1/Read/ReadVariableOp2v/critic/critic/dense_6/kernel/Read/ReadVariableOp7v/critic/critic/dense_6/kernel/Adam/Read/ReadVariableOp9v/critic/critic/dense_6/kernel/Adam_1/Read/ReadVariableOp*<
dtypes2
02.

save_6/control_dependencyIdentitysave_6/ShardedFilename^save_6/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_6/ShardedFilename
£
-save_6/MergeV2Checkpoints/checkpoint_prefixesPacksave_6/ShardedFilename^save_6/control_dependency*
_output_shapes
:*
T0*

axis *
N

save_6/MergeV2CheckpointsMergeV2Checkpoints-save_6/MergeV2Checkpoints/checkpoint_prefixessave_6/Const*
delete_old_dirs(

save_6/IdentityIdentitysave_6/Const^save_6/MergeV2Checkpoints^save_6/control_dependency*
_output_shapes
: *
T0
¹
save_6/RestoreV2/tensor_namesConst*ē
valueŻBŚ.Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/actor/actor/dense/biasBpi/actor/actor/dense/bias/AdamB pi/actor/actor/dense/bias/Adam_1Bpi/actor/actor/dense/kernelB pi/actor/actor/dense/kernel/AdamB"pi/actor/actor/dense/kernel/Adam_1Bpi/actor/actor/dense_1/biasB pi/actor/actor/dense_1/bias/AdamB"pi/actor/actor/dense_1/bias/Adam_1Bpi/actor/actor/dense_1/kernelB"pi/actor/actor/dense_1/kernel/AdamB$pi/actor/actor/dense_1/kernel/Adam_1Bpi/actor/actor/dense_2/biasB pi/actor/actor/dense_2/bias/AdamB"pi/actor/actor/dense_2/bias/Adam_1Bpi/actor/actor/dense_2/kernelB"pi/actor/actor/dense_2/kernel/AdamB$pi/actor/actor/dense_2/kernel/Adam_1Bpi/actor/actor/dense_3/biasB pi/actor/actor/dense_3/bias/AdamB"pi/actor/actor/dense_3/bias/Adam_1Bpi/actor/actor/dense_3/kernelB"pi/actor/actor/dense_3/kernel/AdamB$pi/actor/actor/dense_3/kernel/Adam_1Bv/critic/critic/dense_4/biasB!v/critic/critic/dense_4/bias/AdamB#v/critic/critic/dense_4/bias/Adam_1Bv/critic/critic/dense_4/kernelB#v/critic/critic/dense_4/kernel/AdamB%v/critic/critic/dense_4/kernel/Adam_1Bv/critic/critic/dense_5/biasB!v/critic/critic/dense_5/bias/AdamB#v/critic/critic/dense_5/bias/Adam_1Bv/critic/critic/dense_5/kernelB#v/critic/critic/dense_5/kernel/AdamB%v/critic/critic/dense_5/kernel/Adam_1Bv/critic/critic/dense_6/biasB!v/critic/critic/dense_6/bias/AdamB#v/critic/critic/dense_6/bias/Adam_1Bv/critic/critic/dense_6/kernelB#v/critic/critic/dense_6/kernel/AdamB%v/critic/critic/dense_6/kernel/Adam_1*
dtype0*
_output_shapes
:.
Ä
!save_6/RestoreV2/shape_and_slicesConst*
_output_shapes
:.*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ü
save_6/RestoreV2	RestoreV2save_6/Constsave_6/RestoreV2/tensor_names!save_6/RestoreV2/shape_and_slices*<
dtypes2
02.*Ī
_output_shapes»
ø::::::::::::::::::::::::::::::::::::::::::::::
R
save_6/Identity_1Identitysave_6/RestoreV2*
_output_shapes
:*
T0
X
save_6/AssignVariableOpAssignVariableOpbeta1_powersave_6/Identity_1*
dtype0
T
save_6/Identity_2Identitysave_6/RestoreV2:1*
T0*
_output_shapes
:
\
save_6/AssignVariableOp_1AssignVariableOpbeta1_power_1save_6/Identity_2*
dtype0
T
save_6/Identity_3Identitysave_6/RestoreV2:2*
T0*
_output_shapes
:
Z
save_6/AssignVariableOp_2AssignVariableOpbeta2_powersave_6/Identity_3*
dtype0
T
save_6/Identity_4Identitysave_6/RestoreV2:3*
T0*
_output_shapes
:
\
save_6/AssignVariableOp_3AssignVariableOpbeta2_power_1save_6/Identity_4*
dtype0
T
save_6/Identity_5Identitysave_6/RestoreV2:4*
_output_shapes
:*
T0
h
save_6/AssignVariableOp_4AssignVariableOppi/actor/actor/dense/biassave_6/Identity_5*
dtype0
T
save_6/Identity_6Identitysave_6/RestoreV2:5*
T0*
_output_shapes
:
m
save_6/AssignVariableOp_5AssignVariableOppi/actor/actor/dense/bias/Adamsave_6/Identity_6*
dtype0
T
save_6/Identity_7Identitysave_6/RestoreV2:6*
T0*
_output_shapes
:
o
save_6/AssignVariableOp_6AssignVariableOp pi/actor/actor/dense/bias/Adam_1save_6/Identity_7*
dtype0
T
save_6/Identity_8Identitysave_6/RestoreV2:7*
_output_shapes
:*
T0
j
save_6/AssignVariableOp_7AssignVariableOppi/actor/actor/dense/kernelsave_6/Identity_8*
dtype0
T
save_6/Identity_9Identitysave_6/RestoreV2:8*
_output_shapes
:*
T0
o
save_6/AssignVariableOp_8AssignVariableOp pi/actor/actor/dense/kernel/Adamsave_6/Identity_9*
dtype0
U
save_6/Identity_10Identitysave_6/RestoreV2:9*
_output_shapes
:*
T0
r
save_6/AssignVariableOp_9AssignVariableOp"pi/actor/actor/dense/kernel/Adam_1save_6/Identity_10*
dtype0
V
save_6/Identity_11Identitysave_6/RestoreV2:10*
T0*
_output_shapes
:
l
save_6/AssignVariableOp_10AssignVariableOppi/actor/actor/dense_1/biassave_6/Identity_11*
dtype0
V
save_6/Identity_12Identitysave_6/RestoreV2:11*
T0*
_output_shapes
:
q
save_6/AssignVariableOp_11AssignVariableOp pi/actor/actor/dense_1/bias/Adamsave_6/Identity_12*
dtype0
V
save_6/Identity_13Identitysave_6/RestoreV2:12*
_output_shapes
:*
T0
s
save_6/AssignVariableOp_12AssignVariableOp"pi/actor/actor/dense_1/bias/Adam_1save_6/Identity_13*
dtype0
V
save_6/Identity_14Identitysave_6/RestoreV2:13*
T0*
_output_shapes
:
n
save_6/AssignVariableOp_13AssignVariableOppi/actor/actor/dense_1/kernelsave_6/Identity_14*
dtype0
V
save_6/Identity_15Identitysave_6/RestoreV2:14*
T0*
_output_shapes
:
s
save_6/AssignVariableOp_14AssignVariableOp"pi/actor/actor/dense_1/kernel/Adamsave_6/Identity_15*
dtype0
V
save_6/Identity_16Identitysave_6/RestoreV2:15*
T0*
_output_shapes
:
u
save_6/AssignVariableOp_15AssignVariableOp$pi/actor/actor/dense_1/kernel/Adam_1save_6/Identity_16*
dtype0
V
save_6/Identity_17Identitysave_6/RestoreV2:16*
T0*
_output_shapes
:
l
save_6/AssignVariableOp_16AssignVariableOppi/actor/actor/dense_2/biassave_6/Identity_17*
dtype0
V
save_6/Identity_18Identitysave_6/RestoreV2:17*
T0*
_output_shapes
:
q
save_6/AssignVariableOp_17AssignVariableOp pi/actor/actor/dense_2/bias/Adamsave_6/Identity_18*
dtype0
V
save_6/Identity_19Identitysave_6/RestoreV2:18*
T0*
_output_shapes
:
s
save_6/AssignVariableOp_18AssignVariableOp"pi/actor/actor/dense_2/bias/Adam_1save_6/Identity_19*
dtype0
V
save_6/Identity_20Identitysave_6/RestoreV2:19*
T0*
_output_shapes
:
n
save_6/AssignVariableOp_19AssignVariableOppi/actor/actor/dense_2/kernelsave_6/Identity_20*
dtype0
V
save_6/Identity_21Identitysave_6/RestoreV2:20*
_output_shapes
:*
T0
s
save_6/AssignVariableOp_20AssignVariableOp"pi/actor/actor/dense_2/kernel/Adamsave_6/Identity_21*
dtype0
V
save_6/Identity_22Identitysave_6/RestoreV2:21*
_output_shapes
:*
T0
u
save_6/AssignVariableOp_21AssignVariableOp$pi/actor/actor/dense_2/kernel/Adam_1save_6/Identity_22*
dtype0
V
save_6/Identity_23Identitysave_6/RestoreV2:22*
T0*
_output_shapes
:
l
save_6/AssignVariableOp_22AssignVariableOppi/actor/actor/dense_3/biassave_6/Identity_23*
dtype0
V
save_6/Identity_24Identitysave_6/RestoreV2:23*
_output_shapes
:*
T0
q
save_6/AssignVariableOp_23AssignVariableOp pi/actor/actor/dense_3/bias/Adamsave_6/Identity_24*
dtype0
V
save_6/Identity_25Identitysave_6/RestoreV2:24*
_output_shapes
:*
T0
s
save_6/AssignVariableOp_24AssignVariableOp"pi/actor/actor/dense_3/bias/Adam_1save_6/Identity_25*
dtype0
V
save_6/Identity_26Identitysave_6/RestoreV2:25*
T0*
_output_shapes
:
n
save_6/AssignVariableOp_25AssignVariableOppi/actor/actor/dense_3/kernelsave_6/Identity_26*
dtype0
V
save_6/Identity_27Identitysave_6/RestoreV2:26*
T0*
_output_shapes
:
s
save_6/AssignVariableOp_26AssignVariableOp"pi/actor/actor/dense_3/kernel/Adamsave_6/Identity_27*
dtype0
V
save_6/Identity_28Identitysave_6/RestoreV2:27*
T0*
_output_shapes
:
u
save_6/AssignVariableOp_27AssignVariableOp$pi/actor/actor/dense_3/kernel/Adam_1save_6/Identity_28*
dtype0
V
save_6/Identity_29Identitysave_6/RestoreV2:28*
T0*
_output_shapes
:
m
save_6/AssignVariableOp_28AssignVariableOpv/critic/critic/dense_4/biassave_6/Identity_29*
dtype0
V
save_6/Identity_30Identitysave_6/RestoreV2:29*
T0*
_output_shapes
:
r
save_6/AssignVariableOp_29AssignVariableOp!v/critic/critic/dense_4/bias/Adamsave_6/Identity_30*
dtype0
V
save_6/Identity_31Identitysave_6/RestoreV2:30*
T0*
_output_shapes
:
t
save_6/AssignVariableOp_30AssignVariableOp#v/critic/critic/dense_4/bias/Adam_1save_6/Identity_31*
dtype0
V
save_6/Identity_32Identitysave_6/RestoreV2:31*
_output_shapes
:*
T0
o
save_6/AssignVariableOp_31AssignVariableOpv/critic/critic/dense_4/kernelsave_6/Identity_32*
dtype0
V
save_6/Identity_33Identitysave_6/RestoreV2:32*
T0*
_output_shapes
:
t
save_6/AssignVariableOp_32AssignVariableOp#v/critic/critic/dense_4/kernel/Adamsave_6/Identity_33*
dtype0
V
save_6/Identity_34Identitysave_6/RestoreV2:33*
_output_shapes
:*
T0
v
save_6/AssignVariableOp_33AssignVariableOp%v/critic/critic/dense_4/kernel/Adam_1save_6/Identity_34*
dtype0
V
save_6/Identity_35Identitysave_6/RestoreV2:34*
T0*
_output_shapes
:
m
save_6/AssignVariableOp_34AssignVariableOpv/critic/critic/dense_5/biassave_6/Identity_35*
dtype0
V
save_6/Identity_36Identitysave_6/RestoreV2:35*
T0*
_output_shapes
:
r
save_6/AssignVariableOp_35AssignVariableOp!v/critic/critic/dense_5/bias/Adamsave_6/Identity_36*
dtype0
V
save_6/Identity_37Identitysave_6/RestoreV2:36*
T0*
_output_shapes
:
t
save_6/AssignVariableOp_36AssignVariableOp#v/critic/critic/dense_5/bias/Adam_1save_6/Identity_37*
dtype0
V
save_6/Identity_38Identitysave_6/RestoreV2:37*
T0*
_output_shapes
:
o
save_6/AssignVariableOp_37AssignVariableOpv/critic/critic/dense_5/kernelsave_6/Identity_38*
dtype0
V
save_6/Identity_39Identitysave_6/RestoreV2:38*
_output_shapes
:*
T0
t
save_6/AssignVariableOp_38AssignVariableOp#v/critic/critic/dense_5/kernel/Adamsave_6/Identity_39*
dtype0
V
save_6/Identity_40Identitysave_6/RestoreV2:39*
T0*
_output_shapes
:
v
save_6/AssignVariableOp_39AssignVariableOp%v/critic/critic/dense_5/kernel/Adam_1save_6/Identity_40*
dtype0
V
save_6/Identity_41Identitysave_6/RestoreV2:40*
T0*
_output_shapes
:
m
save_6/AssignVariableOp_40AssignVariableOpv/critic/critic/dense_6/biassave_6/Identity_41*
dtype0
V
save_6/Identity_42Identitysave_6/RestoreV2:41*
T0*
_output_shapes
:
r
save_6/AssignVariableOp_41AssignVariableOp!v/critic/critic/dense_6/bias/Adamsave_6/Identity_42*
dtype0
V
save_6/Identity_43Identitysave_6/RestoreV2:42*
T0*
_output_shapes
:
t
save_6/AssignVariableOp_42AssignVariableOp#v/critic/critic/dense_6/bias/Adam_1save_6/Identity_43*
dtype0
V
save_6/Identity_44Identitysave_6/RestoreV2:43*
_output_shapes
:*
T0
o
save_6/AssignVariableOp_43AssignVariableOpv/critic/critic/dense_6/kernelsave_6/Identity_44*
dtype0
V
save_6/Identity_45Identitysave_6/RestoreV2:44*
_output_shapes
:*
T0
t
save_6/AssignVariableOp_44AssignVariableOp#v/critic/critic/dense_6/kernel/Adamsave_6/Identity_45*
dtype0
V
save_6/Identity_46Identitysave_6/RestoreV2:45*
T0*
_output_shapes
:
v
save_6/AssignVariableOp_45AssignVariableOp%v/critic/critic/dense_6/kernel/Adam_1save_6/Identity_46*
dtype0
Ę

save_6/restore_shardNoOp^save_6/AssignVariableOp^save_6/AssignVariableOp_1^save_6/AssignVariableOp_10^save_6/AssignVariableOp_11^save_6/AssignVariableOp_12^save_6/AssignVariableOp_13^save_6/AssignVariableOp_14^save_6/AssignVariableOp_15^save_6/AssignVariableOp_16^save_6/AssignVariableOp_17^save_6/AssignVariableOp_18^save_6/AssignVariableOp_19^save_6/AssignVariableOp_2^save_6/AssignVariableOp_20^save_6/AssignVariableOp_21^save_6/AssignVariableOp_22^save_6/AssignVariableOp_23^save_6/AssignVariableOp_24^save_6/AssignVariableOp_25^save_6/AssignVariableOp_26^save_6/AssignVariableOp_27^save_6/AssignVariableOp_28^save_6/AssignVariableOp_29^save_6/AssignVariableOp_3^save_6/AssignVariableOp_30^save_6/AssignVariableOp_31^save_6/AssignVariableOp_32^save_6/AssignVariableOp_33^save_6/AssignVariableOp_34^save_6/AssignVariableOp_35^save_6/AssignVariableOp_36^save_6/AssignVariableOp_37^save_6/AssignVariableOp_38^save_6/AssignVariableOp_39^save_6/AssignVariableOp_4^save_6/AssignVariableOp_40^save_6/AssignVariableOp_41^save_6/AssignVariableOp_42^save_6/AssignVariableOp_43^save_6/AssignVariableOp_44^save_6/AssignVariableOp_45^save_6/AssignVariableOp_5^save_6/AssignVariableOp_6^save_6/AssignVariableOp_7^save_6/AssignVariableOp_8^save_6/AssignVariableOp_9
1
save_6/restore_allNoOp^save_6/restore_shard "&B
save_6/Const:0save_6/Identity:0save_6/restore_all (5 @F8"
train_op

Adam
Adam_1"ąC
	variablesŅCĻC
“
pi/actor/actor/dense/kernel:0"pi/actor/actor/dense/kernel/Assign1pi/actor/actor/dense/kernel/Read/ReadVariableOp:0(28pi/actor/actor/dense/kernel/Initializer/random_uniform:08
£
pi/actor/actor/dense/bias:0 pi/actor/actor/dense/bias/Assign/pi/actor/actor/dense/bias/Read/ReadVariableOp:0(2-pi/actor/actor/dense/bias/Initializer/zeros:08
¼
pi/actor/actor/dense_1/kernel:0$pi/actor/actor/dense_1/kernel/Assign3pi/actor/actor/dense_1/kernel/Read/ReadVariableOp:0(2:pi/actor/actor/dense_1/kernel/Initializer/random_uniform:08
«
pi/actor/actor/dense_1/bias:0"pi/actor/actor/dense_1/bias/Assign1pi/actor/actor/dense_1/bias/Read/ReadVariableOp:0(2/pi/actor/actor/dense_1/bias/Initializer/zeros:08
¼
pi/actor/actor/dense_2/kernel:0$pi/actor/actor/dense_2/kernel/Assign3pi/actor/actor/dense_2/kernel/Read/ReadVariableOp:0(2:pi/actor/actor/dense_2/kernel/Initializer/random_uniform:08
«
pi/actor/actor/dense_2/bias:0"pi/actor/actor/dense_2/bias/Assign1pi/actor/actor/dense_2/bias/Read/ReadVariableOp:0(2/pi/actor/actor/dense_2/bias/Initializer/zeros:08
¼
pi/actor/actor/dense_3/kernel:0$pi/actor/actor/dense_3/kernel/Assign3pi/actor/actor/dense_3/kernel/Read/ReadVariableOp:0(2:pi/actor/actor/dense_3/kernel/Initializer/random_uniform:08
«
pi/actor/actor/dense_3/bias:0"pi/actor/actor/dense_3/bias/Assign1pi/actor/actor/dense_3/bias/Read/ReadVariableOp:0(2/pi/actor/actor/dense_3/bias/Initializer/zeros:08
Ą
 v/critic/critic/dense_4/kernel:0%v/critic/critic/dense_4/kernel/Assign4v/critic/critic/dense_4/kernel/Read/ReadVariableOp:0(2;v/critic/critic/dense_4/kernel/Initializer/random_uniform:08
Æ
v/critic/critic/dense_4/bias:0#v/critic/critic/dense_4/bias/Assign2v/critic/critic/dense_4/bias/Read/ReadVariableOp:0(20v/critic/critic/dense_4/bias/Initializer/zeros:08
Ą
 v/critic/critic/dense_5/kernel:0%v/critic/critic/dense_5/kernel/Assign4v/critic/critic/dense_5/kernel/Read/ReadVariableOp:0(2;v/critic/critic/dense_5/kernel/Initializer/random_uniform:08
Æ
v/critic/critic/dense_5/bias:0#v/critic/critic/dense_5/bias/Assign2v/critic/critic/dense_5/bias/Read/ReadVariableOp:0(20v/critic/critic/dense_5/bias/Initializer/zeros:08
Ą
 v/critic/critic/dense_6/kernel:0%v/critic/critic/dense_6/kernel/Assign4v/critic/critic/dense_6/kernel/Read/ReadVariableOp:0(2;v/critic/critic/dense_6/kernel/Initializer/random_uniform:08
Æ
v/critic/critic/dense_6/bias:0#v/critic/critic/dense_6/bias/Assign2v/critic/critic/dense_6/bias/Read/ReadVariableOp:0(20v/critic/critic/dense_6/bias/Initializer/zeros:08
q
beta1_power:0beta1_power/Assign!beta1_power/Read/ReadVariableOp:0(2'beta1_power/Initializer/initial_value:0
q
beta2_power:0beta2_power/Assign!beta2_power/Read/ReadVariableOp:0(2'beta2_power/Initializer/initial_value:0
½
"pi/actor/actor/dense/kernel/Adam:0'pi/actor/actor/dense/kernel/Adam/Assign6pi/actor/actor/dense/kernel/Adam/Read/ReadVariableOp:0(24pi/actor/actor/dense/kernel/Adam/Initializer/zeros:0
Å
$pi/actor/actor/dense/kernel/Adam_1:0)pi/actor/actor/dense/kernel/Adam_1/Assign8pi/actor/actor/dense/kernel/Adam_1/Read/ReadVariableOp:0(26pi/actor/actor/dense/kernel/Adam_1/Initializer/zeros:0
µ
 pi/actor/actor/dense/bias/Adam:0%pi/actor/actor/dense/bias/Adam/Assign4pi/actor/actor/dense/bias/Adam/Read/ReadVariableOp:0(22pi/actor/actor/dense/bias/Adam/Initializer/zeros:0
½
"pi/actor/actor/dense/bias/Adam_1:0'pi/actor/actor/dense/bias/Adam_1/Assign6pi/actor/actor/dense/bias/Adam_1/Read/ReadVariableOp:0(24pi/actor/actor/dense/bias/Adam_1/Initializer/zeros:0
Å
$pi/actor/actor/dense_1/kernel/Adam:0)pi/actor/actor/dense_1/kernel/Adam/Assign8pi/actor/actor/dense_1/kernel/Adam/Read/ReadVariableOp:0(26pi/actor/actor/dense_1/kernel/Adam/Initializer/zeros:0
Ķ
&pi/actor/actor/dense_1/kernel/Adam_1:0+pi/actor/actor/dense_1/kernel/Adam_1/Assign:pi/actor/actor/dense_1/kernel/Adam_1/Read/ReadVariableOp:0(28pi/actor/actor/dense_1/kernel/Adam_1/Initializer/zeros:0
½
"pi/actor/actor/dense_1/bias/Adam:0'pi/actor/actor/dense_1/bias/Adam/Assign6pi/actor/actor/dense_1/bias/Adam/Read/ReadVariableOp:0(24pi/actor/actor/dense_1/bias/Adam/Initializer/zeros:0
Å
$pi/actor/actor/dense_1/bias/Adam_1:0)pi/actor/actor/dense_1/bias/Adam_1/Assign8pi/actor/actor/dense_1/bias/Adam_1/Read/ReadVariableOp:0(26pi/actor/actor/dense_1/bias/Adam_1/Initializer/zeros:0
Å
$pi/actor/actor/dense_2/kernel/Adam:0)pi/actor/actor/dense_2/kernel/Adam/Assign8pi/actor/actor/dense_2/kernel/Adam/Read/ReadVariableOp:0(26pi/actor/actor/dense_2/kernel/Adam/Initializer/zeros:0
Ķ
&pi/actor/actor/dense_2/kernel/Adam_1:0+pi/actor/actor/dense_2/kernel/Adam_1/Assign:pi/actor/actor/dense_2/kernel/Adam_1/Read/ReadVariableOp:0(28pi/actor/actor/dense_2/kernel/Adam_1/Initializer/zeros:0
½
"pi/actor/actor/dense_2/bias/Adam:0'pi/actor/actor/dense_2/bias/Adam/Assign6pi/actor/actor/dense_2/bias/Adam/Read/ReadVariableOp:0(24pi/actor/actor/dense_2/bias/Adam/Initializer/zeros:0
Å
$pi/actor/actor/dense_2/bias/Adam_1:0)pi/actor/actor/dense_2/bias/Adam_1/Assign8pi/actor/actor/dense_2/bias/Adam_1/Read/ReadVariableOp:0(26pi/actor/actor/dense_2/bias/Adam_1/Initializer/zeros:0
Å
$pi/actor/actor/dense_3/kernel/Adam:0)pi/actor/actor/dense_3/kernel/Adam/Assign8pi/actor/actor/dense_3/kernel/Adam/Read/ReadVariableOp:0(26pi/actor/actor/dense_3/kernel/Adam/Initializer/zeros:0
Ķ
&pi/actor/actor/dense_3/kernel/Adam_1:0+pi/actor/actor/dense_3/kernel/Adam_1/Assign:pi/actor/actor/dense_3/kernel/Adam_1/Read/ReadVariableOp:0(28pi/actor/actor/dense_3/kernel/Adam_1/Initializer/zeros:0
½
"pi/actor/actor/dense_3/bias/Adam:0'pi/actor/actor/dense_3/bias/Adam/Assign6pi/actor/actor/dense_3/bias/Adam/Read/ReadVariableOp:0(24pi/actor/actor/dense_3/bias/Adam/Initializer/zeros:0
Å
$pi/actor/actor/dense_3/bias/Adam_1:0)pi/actor/actor/dense_3/bias/Adam_1/Assign8pi/actor/actor/dense_3/bias/Adam_1/Read/ReadVariableOp:0(26pi/actor/actor/dense_3/bias/Adam_1/Initializer/zeros:0
y
beta1_power_1:0beta1_power_1/Assign#beta1_power_1/Read/ReadVariableOp:0(2)beta1_power_1/Initializer/initial_value:0
y
beta2_power_1:0beta2_power_1/Assign#beta2_power_1/Read/ReadVariableOp:0(2)beta2_power_1/Initializer/initial_value:0
É
%v/critic/critic/dense_4/kernel/Adam:0*v/critic/critic/dense_4/kernel/Adam/Assign9v/critic/critic/dense_4/kernel/Adam/Read/ReadVariableOp:0(27v/critic/critic/dense_4/kernel/Adam/Initializer/zeros:0
Ń
'v/critic/critic/dense_4/kernel/Adam_1:0,v/critic/critic/dense_4/kernel/Adam_1/Assign;v/critic/critic/dense_4/kernel/Adam_1/Read/ReadVariableOp:0(29v/critic/critic/dense_4/kernel/Adam_1/Initializer/zeros:0
Į
#v/critic/critic/dense_4/bias/Adam:0(v/critic/critic/dense_4/bias/Adam/Assign7v/critic/critic/dense_4/bias/Adam/Read/ReadVariableOp:0(25v/critic/critic/dense_4/bias/Adam/Initializer/zeros:0
É
%v/critic/critic/dense_4/bias/Adam_1:0*v/critic/critic/dense_4/bias/Adam_1/Assign9v/critic/critic/dense_4/bias/Adam_1/Read/ReadVariableOp:0(27v/critic/critic/dense_4/bias/Adam_1/Initializer/zeros:0
É
%v/critic/critic/dense_5/kernel/Adam:0*v/critic/critic/dense_5/kernel/Adam/Assign9v/critic/critic/dense_5/kernel/Adam/Read/ReadVariableOp:0(27v/critic/critic/dense_5/kernel/Adam/Initializer/zeros:0
Ń
'v/critic/critic/dense_5/kernel/Adam_1:0,v/critic/critic/dense_5/kernel/Adam_1/Assign;v/critic/critic/dense_5/kernel/Adam_1/Read/ReadVariableOp:0(29v/critic/critic/dense_5/kernel/Adam_1/Initializer/zeros:0
Į
#v/critic/critic/dense_5/bias/Adam:0(v/critic/critic/dense_5/bias/Adam/Assign7v/critic/critic/dense_5/bias/Adam/Read/ReadVariableOp:0(25v/critic/critic/dense_5/bias/Adam/Initializer/zeros:0
É
%v/critic/critic/dense_5/bias/Adam_1:0*v/critic/critic/dense_5/bias/Adam_1/Assign9v/critic/critic/dense_5/bias/Adam_1/Read/ReadVariableOp:0(27v/critic/critic/dense_5/bias/Adam_1/Initializer/zeros:0
É
%v/critic/critic/dense_6/kernel/Adam:0*v/critic/critic/dense_6/kernel/Adam/Assign9v/critic/critic/dense_6/kernel/Adam/Read/ReadVariableOp:0(27v/critic/critic/dense_6/kernel/Adam/Initializer/zeros:0
Ń
'v/critic/critic/dense_6/kernel/Adam_1:0,v/critic/critic/dense_6/kernel/Adam_1/Assign;v/critic/critic/dense_6/kernel/Adam_1/Read/ReadVariableOp:0(29v/critic/critic/dense_6/kernel/Adam_1/Initializer/zeros:0
Į
#v/critic/critic/dense_6/bias/Adam:0(v/critic/critic/dense_6/bias/Adam/Assign7v/critic/critic/dense_6/bias/Adam/Read/ReadVariableOp:0(25v/critic/critic/dense_6/bias/Adam/Initializer/zeros:0
É
%v/critic/critic/dense_6/bias/Adam_1:0*v/critic/critic/dense_6/bias/Adam_1/Assign9v/critic/critic/dense_6/bias/Adam_1/Read/ReadVariableOp:0(27v/critic/critic/dense_6/bias/Adam_1/Initializer/zeros:0"
trainable_variables
“
pi/actor/actor/dense/kernel:0"pi/actor/actor/dense/kernel/Assign1pi/actor/actor/dense/kernel/Read/ReadVariableOp:0(28pi/actor/actor/dense/kernel/Initializer/random_uniform:08
£
pi/actor/actor/dense/bias:0 pi/actor/actor/dense/bias/Assign/pi/actor/actor/dense/bias/Read/ReadVariableOp:0(2-pi/actor/actor/dense/bias/Initializer/zeros:08
¼
pi/actor/actor/dense_1/kernel:0$pi/actor/actor/dense_1/kernel/Assign3pi/actor/actor/dense_1/kernel/Read/ReadVariableOp:0(2:pi/actor/actor/dense_1/kernel/Initializer/random_uniform:08
«
pi/actor/actor/dense_1/bias:0"pi/actor/actor/dense_1/bias/Assign1pi/actor/actor/dense_1/bias/Read/ReadVariableOp:0(2/pi/actor/actor/dense_1/bias/Initializer/zeros:08
¼
pi/actor/actor/dense_2/kernel:0$pi/actor/actor/dense_2/kernel/Assign3pi/actor/actor/dense_2/kernel/Read/ReadVariableOp:0(2:pi/actor/actor/dense_2/kernel/Initializer/random_uniform:08
«
pi/actor/actor/dense_2/bias:0"pi/actor/actor/dense_2/bias/Assign1pi/actor/actor/dense_2/bias/Read/ReadVariableOp:0(2/pi/actor/actor/dense_2/bias/Initializer/zeros:08
¼
pi/actor/actor/dense_3/kernel:0$pi/actor/actor/dense_3/kernel/Assign3pi/actor/actor/dense_3/kernel/Read/ReadVariableOp:0(2:pi/actor/actor/dense_3/kernel/Initializer/random_uniform:08
«
pi/actor/actor/dense_3/bias:0"pi/actor/actor/dense_3/bias/Assign1pi/actor/actor/dense_3/bias/Read/ReadVariableOp:0(2/pi/actor/actor/dense_3/bias/Initializer/zeros:08
Ą
 v/critic/critic/dense_4/kernel:0%v/critic/critic/dense_4/kernel/Assign4v/critic/critic/dense_4/kernel/Read/ReadVariableOp:0(2;v/critic/critic/dense_4/kernel/Initializer/random_uniform:08
Æ
v/critic/critic/dense_4/bias:0#v/critic/critic/dense_4/bias/Assign2v/critic/critic/dense_4/bias/Read/ReadVariableOp:0(20v/critic/critic/dense_4/bias/Initializer/zeros:08
Ą
 v/critic/critic/dense_5/kernel:0%v/critic/critic/dense_5/kernel/Assign4v/critic/critic/dense_5/kernel/Read/ReadVariableOp:0(2;v/critic/critic/dense_5/kernel/Initializer/random_uniform:08
Æ
v/critic/critic/dense_5/bias:0#v/critic/critic/dense_5/bias/Assign2v/critic/critic/dense_5/bias/Read/ReadVariableOp:0(20v/critic/critic/dense_5/bias/Initializer/zeros:08
Ą
 v/critic/critic/dense_6/kernel:0%v/critic/critic/dense_6/kernel/Assign4v/critic/critic/dense_6/kernel/Read/ReadVariableOp:0(2;v/critic/critic/dense_6/kernel/Initializer/random_uniform:08
Æ
v/critic/critic/dense_6/bias:0#v/critic/critic/dense_6/bias/Assign2v/critic/critic/dense_6/bias/Read/ReadVariableOp:0(20v/critic/critic/dense_6/bias/Initializer/zeros:08*Ø
serving_default
)
x$
Placeholder:0’’’’’’’’’#
v
v/Squeeze:0’’’’’’’’’&
pi 
	pi/Tanh:0’’’’’’’’’tensorflow/serving/predict