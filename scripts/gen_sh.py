#!/usr/bin/env python3

# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from sympy import *
max_level = 8
do_diff = True

init_printing()
S=[0]
C=[1]
x,y,z=symbols('x y z')
x2,y2,z2=symbols('x2 y2 z2')
x4,y4,z4=symbols('x4 y4 z4')
x6,y6,z6=symbols('x6 y6 z6')
xz,yz,xy=symbols('xz yz xy')
for i in range(10):
	S.append(simplify(x*S[i]+y*C[i]))
	C.append(simplify(x*C[i]-y*S[i]))
def K(l,m):
	return sqrt((2*l+1)*factorial(l-abs(m))/(4*pi*factorial(l+abs(m))));
def P(l,m):
	if l==0 and m==0:
		return 1
	if l==m:
		return (1-2*m)*P(m-1,m-1)
	if l==m+1:
		return (2*m+1)*z*P(m,m)
	return (((2*l-1)*z*P(l-1,m)-(l+m-1)*P(l-2,m))/(l-m))
def Y(l,m):
	if m>0:
		return sqrt(2)*K(l,m)*C[m]*P(l,m)
	if m<0:
		return sqrt(2)*K(l,m)*S[-m]*P(l,-m)
	return K(l,m)*P(l,m)

#-------------------------------------------------------
for diffvar in [False,x,y,z] if do_diff else [False]:
	j=0
	for l in range(max_level):
		if l > 0:
			print(f'if (sh_degree <= {l}) {{ return; }}')
		for m in range(-l,l+1):
			o=simplify(Y(l,m))
			if diffvar:
				o=diff(o,diffvar)
			o=simplify(o)
			o=o.subs(x*x,x2)
			o=o.subs(y*y,y2)
			o=o.subs(z*z,z2)
			o=o.subs(x*y,xy)
			o=o.subs(x*z,xz)
			o=o.subs(y*z,yz)
			o=o.subs(x2*x2*x2,x6)
			o=o.subs(y2*y2*y2,y6)
			o=o.subs(z2*z2*z2,z6)
			o=o.subs(x2*x2,x4)
			o=o.subs(y2*y2,y4)
			o=o.subs(z2*z2,z4)
			o=simplify(o)
			out=f'd{diffvar}' if diffvar else 'out'
			if diffvar:
				print(f'{"// " if o == 0 else ""}d{diffvar} += (float)*grad({j}) * ({ccode(N(o))});\t\t\t\t//', o)
			else:
				print(f'out[{j}] =', ccode(N(o)), ';\t\t\t\t//', o)
			j=j+1
