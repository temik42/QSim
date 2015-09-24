

xs=1200
ys=800
dx=0
dy=0
scale=.3

base=widget_base(XSIZE=xs, YSIZE=ys,/TLB_KILL_REQUEST_EVENTS)
;itwin=obj_new('IDLitWindow', location=[100,0], dimensions=[600,600])
draw=widget_draw(xsize=xs, ysize=ys, xoffset=0, yoffset=50, base, graphics_level=2, retain=2, uvalue='draw',/EXPOSE_EVENTS, /BUTTON_EVENTS)

slider=CW_FSLIDER(base, xsize=100, ysize=0, uvalue='slide', /drag, value=scale, min=0.01, max=10.0)


WIDGET_CONTROL, BASE, /REALIZE
WIN=OBJ_NEW()
WIDGET_CONTROL, DRAW, GET_VALUE=VALUE
WIN=VALUE

WIDGET_CONTROL, Draw, GET_VALUE=win

model=OBJ_NEW('IDLgrModel')
;model= OBJ_NEW('IDLitVisualization')
;manip= OBJ_NEW('IDLitManipulator')
;itwin->AddWindowEventObserver, manip
;manip->OnMouseDown, itwin, p,q, ButtonMask, Modifiers, NumClicks


VIEW=OBJ_NEW('IDLgrView', COLOR=[0,0,0], VIEW=1/scale*[0,0,xs,ys],zclip=[2000,-1000], eye=2000+0.1)







rtr=xs/2
tx=xs/2
ty=ys/2


Track1 = OBJ_NEW('TrackBall', [tx,ty], rtr)
Track2 = OBJ_NEW('TrackBall', [tx,ty], rtr)

;goto, q

n=64l

nz=64l





xi=reform(geni(n,n),n^2)
yi=reform(geni(n,n,/y),n^2)

PARTICLE_TRACE, transpose(field,[3,0,1,2]), transpose([[xi],[yi],[intarr(n^2)]]), Verts, Conn

nconn = n_elements(conn)

i=0

while i lt nconn do begin
nv=conn[i]

ct=extrac(conn, i+1, nv)
vt=verts[*,ct]


if vt[2, nv-1] lt 2 then begin
color=[0,0,255]
field_line=OBJ_NEW('IDLgrPolyline', vt, COLOR=color)
model->add, field_line
endif


i+=nv+1
endwhile


;fline=OBJ_NEW('IDLgrPolyline', verts, polylines=conn, COLOR=[0,255,0])
surf = OBJ_NEW('IDLgrSurface', [[0,0],[0,0]],[[0,n],[0,n]],[[0,0],[n,n]], color=[255,255,255],style=2)

image=OBJ_NEW('IDLgrImage', bytscl(b[*,*,0,2]))
surf->SetProperty, texture_map=image

model->add,surf
;model->add, fline

VIEW->ADD, model

;t3d, matrix=matrix,translate=[-10,-10,0]
t3d, matrix=matrix,scale=[10,10,10]


;t3d, matrix, matrix=matrix,rotate=[-90,-90,0]


model->setproperty, transform=matrix

state={btndown:0b, scale:scale, dx:dx, dy:dy,slider:slider, track1:track1, track2:track2,model:model, Draw: Draw, Win: Win, View: View, rtr:rtr}
WIDGET_CONTROL, Base, SET_UVALUE=State, /NO_COPY
win->draw, view

XMANAGER, 'model', Base, /NO_BLOCK

end

