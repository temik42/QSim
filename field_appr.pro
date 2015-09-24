function pfield1, data

dim=size(data, /dim)
nx=dim[0]
ny=dim[1]

temp = extrac(data, 0, 0, 2*nx, 2*ny)


xi=geni(2*nx,2*ny,/c)
yi=geni(2*nx,2*ny,/c,/y)


u=shift(xi,-nx,-ny)/float(2*nx)
v=shift(yi,-nx,-ny)/float(2*ny)
q=sqrt(u^2+v^2)

fdata=fft(temp)*exp(-2*!pi*q)

b=dblarr(nx,ny,3)

gx0=u*complex(0,-1)/(q>(1./nx/ny))
gy0=v*complex(0,-1)/(q>(1./nx/ny))


b[*,*,0]=extrac(re(fft(fdata*gx0,1,/do)), 0, 0, nx, ny)
b[*,*,1]=extrac(re(fft(fdata*gy0,1,/do)), 0, 0, nx, ny)
b[*,*,2]=extrac(re(fft(fdata,1,/do)), 0, 0, nx, ny)

return, reform(b)
end


function rk2, f, xi, yi
x = xi
y = yi


dim=size(f,/dim)
step = 1
maxi = 200

verts = [x,y]


h = sqrt(f[*,*,0]^2.+f[*,*,1]^2.)
q = h/abs(f[*,*,2])

q1 = interpolate(q,x,y,cubic=-0.5)

i=0
while (q1 gt 0.3) and (i lt maxi) and  $
    (x gt 0) and (x lt dim[0]-1) and (y gt 0) and (y lt dim[1]-1)  do begin


fx = bilinear(f[*,*,0],x,y)
fy = bilinear(f[*,*,1],x,y)

dx = step*fx/sqrt(fx^2+fy^2)
dy = step*fy/sqrt(fx^2+fy^2)
x1 = x+dx
y1 = y+dy

fx1 = 0.5*(fx+bilinear(f[*,*,0],x1,y1))
fy1 = 0.5*(fy+bilinear(f[*,*,1],x1,y1))

dx = step*fx1/sqrt(fx1^2+fy1^2)
dy = step*fy1/sqrt(fx1^2+fy1^2)
x2 = x+dx
y2 = y+dy


x=x2
y=y2


q1 = interpolate(q,x,y,cubic=-0.5)
verts = [[verts], [x,y]]
i++
endwhile


return, verts
end




function pf, qi, ci, bi, ri
nq = n_elements(qi)
nr = n_elements(ri[0,*])

dxi = replicate(1,nr)##reform(ci[0,*]) - reform(ri[0,*])##replicate(1,nq)
dyi = replicate(1,nr)##reform(ci[1,*]) - reform(ri[1,*])##replicate(1,nq)

dri = sqrt(dxi^2+dyi^2)

fxi = dxi/dri^3 ## qi
fyi = dyi/dri^3 ## qi
fi = sqrt(fxi^2 + fyi^2)

fxi /= fi
fyi /= fi

return, (reform(fxi)##bi[0,*]+reform(fyi)##bi[1,*])/nr
end





if ~keyword_set(data) then restore, 'Q:\Dropbox\data\20110225\sdo.x766y1118.20110225.sav'

ci = [[48.1250, 19.7500], $
      [45.1250, 32.5000], $
      [31.8750, 28.1250], $
      [35.5000, 25.7500], $
      [32.2500, 34.0000], $
      [18.6250, 35.7500], $
      [20.8750, 41.3750]]

;k=100
k=545
k1 = k*12./45

temp = total(data.hmi.flux[*,*,k1-4:k1+4],3)/9.
temp = temp*(abs(temp) gt 10)


window, 1, xs=512, ys=512
tvscl, rebin(temp,512,512, /s)

plots, ci[0,*]*8, ci[1,*]*8, psym=1, /dev, color='00ff00'xl
xyouts, ci[0,*]*8, ci[1,*]*8, string(indgen(7),format='(i01)'), /dev


field = pfield1(temp)
;tvscl, rebin(sqrt(total(field[*,*,0:1]^2,3)),512,512, /s)

ri = randomn(seed, 2, 10)
;plots, (ri[0,*])*64+255, ri[1,*]*64+255, psym=1, /dev, color='ff0000'xl

bi = [bilinear(field[*,*,0],ri[0,*],ri[1,*]), $
      bilinear(field[*,*,1],ri[0,*],ri[1,*])]

bi[0,*]/= sqrt(total(bi^2,1))
bi[1,*]/= sqrt(total(bi^2,1))


pos = total(temp>0)
neg = total(temp<0)



q = replicate(1, 7)
q[0:2] = neg/3
q[3:6] = pos/4

dq = replicate(0, 7)
dq[0] = 10

f = pf(q, ci, bi, ri)
df = pf(q+dq, ci, bi, ri) - f

a=(f-1)/df

q1 = q-dq*a[0]






end



for i=0, 31 do for j=0, 31 do begin

verts = rk2(-field,i*2, j*2)
n = n_elements(verts[0,*])

plots, verts[0,*]*8, verts[1,*]*8, /device

verts = rk2(field,i*2, j*2)
n = n_elements(verts[0,*])
plots, verts[0,*]*8, verts[1,*]*8, /device
endfor




end