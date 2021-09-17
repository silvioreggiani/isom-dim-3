def mat_SR(M):
    return matrix(M.nrows(), M.ncols(), lambda i,j: M[i,j].expr())

class non_unimod_dim_3:
    def __init__(self, name, c):
        self.name = name
        self.subindex = c
        self.manifold = Manifold(3, name)
        self.chart = self.manifold.chart(coordinates='x0 x1 x2')
        self.x0 = self.chart[0]
        self.x1 = self.chart[1]
        self.x2 = self.chart[2]
#         self.canonical_frame = self.chart.frame()
        self.zero = self.manifold.point((0,0,0),name='e')
        
    def semidirect_rep(self, t):
        I2 = identity_matrix(2)
        c = self.subindex
        Mc = matrix(2,2,[-1,-c,1,1])
        z = var("z")
        if c == 1:
            phi_ct = exp(t) * (I2 +  t * Mc)
        elif c < 1:
            phi_ct = exp(t) * (cosh(z*t) * I2 +  sinh(z*t) /z * Mc).subs(cosh(z*t)==(exp(z*t)+exp(-z*t))/2).subs(sinh(z*t)==(exp(z*t)-exp(-z*t))/2).subs(z=sqrt(1-c))
            for i in range(phi_ct.nrows()):
                for j in range(phi_ct.ncols()):
                    phi_ct[i,j] = phi_ct[i,j]._sympy_().simplify()._sage_()
        elif c > 1:
            phi_ct = exp(t) * (cosh(z*t) * I2 +  sinh(z*t) /z * Mc).subs(cosh(z*t)==(exp(z*t)+exp(-z*t))/2).subs(sinh(z*t)==(exp(z*t)-exp(-z*t))/2).subs(z=sqrt(1-c))
        return phi_ct.simplify_full()
        
    def left_translation(self, p):
        r"""
        Left translation $x \mapsto px$
        """
        p0,p1,p2 = p.coordinates()
        x0,x1,x2 = self.chart[:]
        M = self.manifold
        p_vect = vector(p.coordinates()[:2])
        x_vect = vector(self.chart[:2])
        normal_part = (p_vect + self.semidirect_rep(p2) * x_vect)
        normal_part0 = sympy.simplify(normal_part[0])
        normal_part1 = sympy.simplify(normal_part[1])
        Lp = M.diffeomorphism(M, [normal_part0, normal_part1, p2+x2])
        return Lp
    
    def right_translation(self, p):
        r"""
        Right translation $x \mapsto xp$
        """
        p0,p1,p2 = p.coordinates()
        x0,x1,x2 = self.chart[:]
        M = self.manifold
        p_vect = vector(p.coordinates()[:2])
        x_vect = vector(self.chart[:2])
        normal_part = (x_vect + self.semidirect_rep(x2) * p_vect)
        normal_part0 = sympy.simplify(normal_part[0])
        normal_part1 = sympy.simplify(normal_part[1])
        Rp = M.diffeomorphism(M, [normal_part0, normal_part1, x2+p2])
        return Rp
    
    def left_inv_vect_field(self,v0,v1,v2,name=None):
        r"""
        Left invariant vector field in the direction of $(v_0, v_1, v_2)$
        """
        RR.<t> = RealLine()
        M = self.manifold
#         X = self.chart[:]
        X = [self.x0, self.x1, self.x2]
        cv = M.curve([t*v0,t*v1,t*v2],t,name="cv")
        c0,c1,c2 = self.right_translation(cv(t))(X[:]).coord()
        dc0 = diff(c0,t).subs(t=0)._sympy_().simplify()._sage_()
        dc1 = diff(c1,t).subs(t=0)._sympy_().simplify()._sage_()
        dc2 = diff(c2,t).subs(t=0)._sympy_().simplify()._sage_()
        V = M.vector_field(dc0,dc1,dc2,name=name)
        return V

    def right_inv_vect_field(self,v0,v1,v2,name=None):
        r"""
        Right invariant vector field in the direction of $(v_0, v_1, v_2)$
        """
        RR.<t> = RealLine()
        M = self.manifold
        X = self.chart[:]
        cv = M.curve([t*v0,t*v1,t*v2],t,name="cv")
        c0,c1,c2 = self.left_translation(cv(t))(X[:]).coord()
        c0 = c0._sympy_().simplify()._sage_()
        c1 = c1._sympy_().simplify()._sage_()
        c2 = c2._sympy_().simplify()._sage_()
        dc0 = diff(c0,t).subs(t=0)
        dc1 = diff(c1,t).subs(t=0)
        dc2 = diff(c2,t).subs(t=0)
        V = M.vector_field(dc0,dc1,dc2,name=name)
        return V
    
    def frame_left_inv(self):
        L0 = self.left_inv_vect_field(1,0,0,r'L_0')
        L1 = self.left_inv_vect_field(0,1,0,r'L_1')
        L2 = self.left_inv_vect_field(0,0,1,r'L_2')
        frame_left_inv = self.manifold.vector_frame('L',(L0,L1,L2))
        return frame_left_inv
    
    def frame_right_inv(self):
        R0 = self.right_inv_vect_field(1,0,0,r'R_0')
        R1 = self.right_inv_vect_field(0,1,0,r'R_1')
        R2 = self.right_inv_vect_field(0,0,1,r'R_2')
        frame_right_inv = self.manifold.vector_frame('R',(R0,R1,R2))
        return frame_right_inv
    
    def left_inv_metric(self, coefs, name=None):
        row1 = coefs[:3]
        row2 = coefs[3:6]
        row3 = coefs[6:]
        frame = self.frame_left_inv()
        g_tensor = self.manifold.tensor_field(0, 2, [row1, row2, row3],frame=frame,name=name)
        g = self.manifold.metric(name=name)
        g[:] = g_tensor[:]
        return g
