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


function rk2, f, x, y
dim=size(f,/dim)
step = 1
maxi = 200

verts = [x,y]


h = sqrt(f[*,*,0]^2.+f[*,*,1]^2.)
q = h/(abs(f[*,*,2])>1)

q1 = interpolate(q,x,y,cubic=-0.5)

i=0
while (q1 gt 0.3) and (i lt maxi) and  $
    (x gt 3) and (x lt dim[0]-4) and (y gt 3) and (y lt dim[1]-4)  do begin


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


if q1 lt 0.3 then return, verts else return, [-1,-1]
end




if ~keyword_set(data) then restore, 'Q:\Dropbox\data\20110225\sdo.x766y1118.20110225.sav'

;k=100
k=545

window, 4, xs=512, ys=512
tvscl, rebin(data.aia_193.flux[*,*,k],512,512, /s)

k1 = k*12./45
temp = total(data.hmi.flux[*,*,k1-4:k1+4],3)/9.
temp = temp*(abs(temp) gt 50)





field = pfield1(temp)
;tvscl, rebin(sqrt(total(field[*,*,0:1]^2,3)),512,512, /s)
window, 1, xs=512, ys=512
tvscl, rebin(field[*,*,2],512,512, /s)

;end



for i=0, 31 do for j=0, 31 do begin

verts = rk2(-field,2*i, 2*j)
nv = n_elements(verts[0,*])
;if nv gt 1 then begin
;plots, verts[0,nv-1]*8,verts[1,nv-1]*8, /dev, psym=3, color='0000ff'xl
;endif
plots, verts[0,*]*8, verts[1,*]*8, /device

verts = rk2(field,2*i, 2*j)
nv = n_elements(verts[0,*])
;if nv gt 1 then begin
;plots, verts[0,nv-1]*8,verts[1,nv-1]*8, /dev, psym=3, color='ff0000'xl
;endif
plots, verts[0,*]*8, verts[1,*]*8, /device
endfor




end