"""This file summarizes functionality for the construction of ConstraintNet with
output-constraints in form of convex polytopes. It is possible to constrain
output-parts to different convex polytopes independently.

The functionality for modelling the output-constraints consists namely of: 

    - Functors to create a tensor representation g(s) of the constraint
        parameter s. The functor can be selected via the option
        opts.opts2constr_para_repr. E.g. opts.opts2constr_para_repr =
        'opts2const_feat_planes' would call the function opts2const_feat_planes
        which instantiate the functor ConstFeatPlanes. ConstFeatPlanes is then
        used for g(s). 
    - Functors to tranform the constraint parameter s into a vertex
        representation. The vertex representation consists of vertices which
        describe the convex polytope(s). The functor can be selected via the option
        opts.opts2constr_para_trf. E.g. opts.opts2constr_para_trf = 'opts2v_polys_bb' would call
        the function opts2v_polys_bb which instantiate the functor VPolysBB.
        VPolysBB creates the vertice representation for bounding box
        constraints. 
        We define the following format for the vertex representation v_polys: 
        v_polys (list): [out_part_1, out_part_2, ...]
            v_polys is a list with length equal to the number of output parts
            which should be independently constrained. Each list element in
            v_polys corresponds to one output part.
        out_part_i (list): [v_convex_poly_1]
            Each out_part_i in v_polys is a list of length 1 and the element
            corresponds to the vertice representation for the convex polytope of
            the constraint for this part. In future this list could be longer
            than one to model non convex polytopes by a set of convex polytopes.
        v_convex_poly_1 (torch tensor): shape (N, n_v, dim_v)
            The vertice representation for a convex polytope is given by a torch
            tensor with shape (N, n_v, dim_v). N is the batch size, n_v the
            number of vertices and dim_v the dimension of the vertices. The
            entries are given by the coordinates of the vertices.
    - PyTorch modules for the constraint-guard layer, i.e. the mapping from the
        intermediate representation z to the constrained output region. The
        module can be selected via the option opts.opts2constr_guard_layer. For
        the considered convex polytope constraints in this file, the PyTorch
        module "Polys" can be selected via opts.opts2constr_guard_layer =
        'opts2polys'. Polys constrains different output parts to different
        convex polytopes. For each output part, the number of vertices of the
        convex polytope must be added to  opts.polys_convex_polys_v_n, the
        dimension of the vertices to opts.polys_convex_polys_v_dim and a 1 to
        opts.polys_output_parts (in future non-convex polytopes might be defined
        and then this number would be the number of convex polytopes it consists
        of). E.g. consider an output-constraint consisting of three independent
        constraints for three output parts. Furthermore, the constraint for the
        first part is a convex polytope in 1d with 2 vertices, the constraint
        for the second part is a convex polytope in 2d with 3 vertices and the
        constraint for the third part is a convex polytope in 3d with 5
        vertices. Then the options should be set to
        opts.polys_convex_polys_v_n = [2, 3, 5]
        opts.polys_convex_polys_v_dim = [1, 2, 3]
        opts.polys_output_parts = [1, 1, 1]

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def opts2const_feat_planes(opts):
    """Creates ConstFeatPlanes functor by calling its constructor with 
    options from opts.

    Args:
        opts (obj): Namespace object with options.

    Returns:
        const_feat_planes (obj): Instantiated ConstFeatPlanes functor.
    """
    return ConstFeatPlanes(
            opts.const_feat_planes_h,
            opts.const_feat_planes_w,
            opts.const_feat_planes_n_channels,
            opts.const_feat_planes_repeat_channels,
            opts.const_feat_planes_norm_factor
            )


class ConstFeatPlanes:
    """This functor generates a tensor representation g(s) of the constraint
    parameter s. 

    Each component of the constraint parameter s corresponds to
    <repeat_channels> channels of the generated tensor. The assigned channels
    have entries with a constant value and are given by a constraint parameter
    rescaled with a factor. The height and width of the tensor can be specified.
    """
        
    def __init__(self, h, w, n_channels, repeat_channels=1, norm_factor=1.):
        """Initialization for setting parameters.

        Args:
            h (int): Height of channels.
            w (int): Width of channels.
            n_channels (int): Number of channels of generated tensor. 
                Specified number must match
                n_channels = <length of  constraint parameter) * repeat_channels
            repeat_channels (int): The channel for a constraint parameter can be
                replicated <repeat_channels> times.
            norm_factor (float or list): Factor to normalize the values of the 
                tensor. If float, all constraint parameter components are
                rescaled with this factor. If list, the length of the list must
                match the number of constraint parameter components and the list
                elements must be of type float. Each constraint parameter
                component is then rescaled with the corresponding factor in the
                list.
        """
        self.h = h
        self.w = w
        self.n_channels = n_channels
        
        self.repeat_channels = repeat_channels
        
        #extract number of constraint parameter components
        n_constr_para = int(n_channels / repeat_channels)
        if not n_channels == n_constr_para * repeat_channels:
            raise ValueError('Number of channels in constraint parameter \
                    tensor representation must be a \
                    multiple of repeat_channels. But n_channels={n_channels} \
                    and repeat_channels={repeat_channels}'.format(
                        n_channels = n_channels,
                        repeat_channels = repeat_channels)
                    )
        
        #convert norm_factor scalar in list format
        self.norm_factor = norm_factor

        if isinstance(self.norm_factor, float):
            norm_factor_value = self.norm_factor
            self.norm_factor = []
            for i in range(n_constr_para):
                self.norm_factor.append(norm_factor_value)
        if len(self.norm_factor)==1:
            norm_factor_value = self.norm_factor[0]
            self.norm_factor = []
            for i in range(n_constr_para):
                self.norm_factor.append(norm_factor_value)


        if not len(self.norm_factor) * repeat_channels == n_channels:
            raise ValueError('Number of norm factors for constr_para must \
                    match n_channels / repeat_channels. But \
                    len(norm_factor)={len_norm} is not equal to \
                    n_channels / repeat_channels = {n_channels} / \
                    {repeat_channels}'.format(
                        len_norm = len(self.norm_factor),
                        n_channels = n_channels,
                        repeat_channels = repeat_channels)
                    )


    def __call__(self, constr_para):
        """Functor to create tensor representation g(s).

        Args:
            constr_para (obj): Pytorch tensor of shape (N, n_constr_para)
                which specifies the output-constraint. 

        Returns:
            constr_para_repr (obj): Pytorch tensor for tensor representation
                g(s) of the constraint parameter with shape 
                (N, c_constr_para_repr, H, W).
        """

        if not self.n_channels == constr_para.shape[1] * self.repeat_channels:
            raise ValueError('Number of channels of the tensor representation \
                        of the constraint parameter must match with  \
                        the number of constraint parameter components times \
                        repeat_channel. But n_channels={n_channels} is \
                        not equal to n_constr_para * repeat_channels = \
                        {n_constr_para} * {repeat_channels}'.format(
                            n_channels = self.n_channels,
                            n_constr_para = constr_para.shape[1],
                            repeat_channels = self.repeat_channels)
                        )


        #create region feature tensor of correct shape
        constr_para_repr = constr_para.new(
                constr_para.shape[0], 
                self.n_channels, 
                self.h,
                self.w
                )

        #fill the constr_features tensor with the normed constr_para
        for i, sample in enumerate(constr_para):
            for j, para in enumerate(sample):
                j_in = j * self.repeat_channels
                for l in range(self.repeat_channels):
                    constr_para_repr[i, j_in + l,:,:] = para * self.norm_factor[j]
        
        return constr_para_repr

def opts2v_polys_bb_rel(opts):
    """Creates VPolysBbRel functor by calling its constructor with options
    from opts.

    Args:
        opts (obj): Namespace object with options.

    Returns:
        v_polys_bb_rel (obj): Instantiated VPolysBbRel functor.
    """
    return VPolysBbRel()

class VPolysBbRel:
    """This functor generates the vertex representation v_polys for bounding box
    constraints in combination with constraints for the relative relations (The
    eyes are above the nose and the left eye is in fact left with respect to the
    right eye).
    """
    def __init__(self):
        pass

    def __call__(self, constr_para):
        """The functor gets the constraint parameter and generates the 
        corresponding vertices representation. 
        
        Args:
            constr_para (obj): Torch tensor for the constraint 
                parameter with shape (N, n_constr_para=4). There are 4 
                constraint parameter components, they are ordered in the
                following way (l_x, u_x, l_y, u_y). They encode the positions of
                the boundaries of the bounding box:
                    l_x: left/ lower x
                    u_x: right/ upper x
                    l_y: upper/ lower y (y coordinates start with 0 at the top
                        of the image)
                    u_y: lower/ upper y (y coordinates start with 0 at the top
                        of the image)
                                    
        Returns:
            v_polys (obj): Vertex representation of the constraint parameter.
                The output y for the neural network is ordered in the following
                way: (x_nose, x_lefteye, y_righteye, y_lefteye, y_righteye,
                y_nose)
        """
        #1d polytope for x_nose
        #shape poly_1d (N, n_v=2, dim_v=1), dim_vertices: x_nose
        poly_1d = constr_para.new(constr_para.shape[0], 2, 1) 
        #v_1 = (l_x)
        poly_1d[:, 0, 0] = constr_para[:, 0]
        #v_2 = (u_x)
        poly_1d[:, 1, 0] = constr_para[:, 1]

        #2d polytope for x_lefteye and x_righteye
        #shape (N, n_v=3, dim_v=2)
        #dim_vertices: x_lefteye, x_righteye
        poly_2d = constr_para.new(constr_para.shape[0], 3, 2)
        #v_1 = (l_x, l_x)
        poly_2d[:, 0, 0] = constr_para[:, 0]
        poly_2d[:, 0, 1] = constr_para[:, 0]
        #v_2 = (l_x, u_x)
        poly_2d[:, 1, 0] = constr_para[:, 0]
        poly_2d[:, 1, 1] = constr_para[:, 1]
        #v_3 = (u_x, u_x)
        poly_2d[:, 2, 0] = constr_para[:, 1]
        poly_2d[:, 2, 1] = constr_para[:, 1]

        #3-d polytope for y_lefteye, y_righteye and y_nose
        #shape (N, n_v=5, dim_v=3)
        #dim_vertices: y_lefteye, y_righteye, y_nose
        poly_3d = constr_para.new(constr_para.shape[0], 5, 3)
        #v_1 = (l_y, l_y, l_y)
        poly_3d[:, 0, 0] = constr_para[:, 2]
        poly_3d[:, 0, 1] = constr_para[:, 2]
        poly_3d[:, 0, 2] = constr_para[:, 2]
        #v_2 = (l_y, l_y, u_y)
        poly_3d[:, 1, 0] = constr_para[:, 2]
        poly_3d[:, 1, 1] = constr_para[:, 2]
        poly_3d[:, 1, 2] = constr_para[:, 3]
        #v_3 = (l_y, u_y, u_y) 
        poly_3d[:, 2, 0] = constr_para[:, 2]
        poly_3d[:, 2, 1] = constr_para[:, 3]
        poly_3d[:, 2, 2] = constr_para[:, 3]
        #v_4 = (u_y, u_y, u_y) 
        poly_3d[:, 3, 0] = constr_para[:, 3]
        poly_3d[:, 3, 1] = constr_para[:, 3]
        poly_3d[:, 3, 2] = constr_para[:, 3]
        #v_5 = (u_y, l_y, u_y) 
        poly_3d[:, 4, 0] = constr_para[:, 3]
        poly_3d[:, 4, 1] = constr_para[:, 2]
        poly_3d[:, 4, 2] = constr_para[:, 3]

        v_polys = [[poly_1d,], [poly_2d,], [poly_3d,]]
        
        return v_polys

def opts2v_polys_bb(opts):
    """Creates VPolysBb functor by calling its constructor with options 
    from opts.

    Args:
        opts (obj): Namespace object with options.

    Returns:
        v_polys_bb (obj): Instantiated VPolysBb functor.
    """
    return VPolysBb(opts.lm_ordering_lm_order)


class VPolysBb:
    """This functor generates the vertex representation v_polys for bounding
    box constraints, i.e. the landmarks for left eye, right eye and nose are
    constrained to a bounding box.
    """

    def __init__(self, lm_ordering_lm_order):
        """Initialization.

        Args:
            lm_ordering_lm_order (list): Order of the landmarks for the output
                of the neural network. E.g. ['nose_x', 'lefteye_x', ... ].
        """
        self.lm_ordering_lm_order = lm_ordering_lm_order

    def __call__(self, constr_para):
        """The functor gets the constraint parameter and generates the
        vertex representation.

        Args:
            constr_para (obj): Torch tensor containing the constraint
                parameters with shape (N, n_constr_para=4). There are 4
                constraint parameters, they are ordered in the following way
                (l_x, u_x, l_y, u_y). They encode the positions of the
                boundaries of the bounding box of the face detector:
                    l_x: left/ lower x
                    u_x: right/ upper x
                    l_y: upper/ lower y (y coordinates start with 0 at the top
                        of the image)
                    u_y: lower/ upper y (y coordinates start with 0 at the top
                        of the image)

        Returns:
            v_polys (obj): Vertice representation of the constraint parameters.
        """
        v_polys = []
        for lm in self.lm_ordering_lm_order:
            if '_x' in lm:
                poly_1d = constr_para.new(constr_para.shape[0], 2, 1)
                # v_1 = (l_x)
                poly_1d[:, 0, 0] = constr_para[:, 0]
                # v_2 = (u_x)
                poly_1d[:, 1, 0] = constr_para[:, 1]
            elif '_y' in lm:
                poly_1d = constr_para.new(constr_para.shape[0], 2, 1)
                # v_1 = (l_y)
                poly_1d[:, 0, 0] = constr_para[:, 2]
                # v_2 = (u_y)
                poly_1d[:, 1, 0] = constr_para[:, 3]
            v_polys.append([poly_1d])

        return v_polys

def opts2v_polys_2d_convex_poly(opts):
    """
    parameters from opts.

    Args:
        opts (obj): Namespace object returned by parser with settings.

    Returns:
        opts2v_polys_lm_xy (obj): Instantiated VPolysLmXY functor.
    """
    return VPolys2DConvexPoly()


class VPolys2DConvexPoly:
    """This functor generates the vertex representation v_polys for constraints
    in form of one 2d-convex polytope.
    
    We assume that the constraint parameter consists of concatenated vertices
    coordinates (x0,y0, ..., xN,yN) of the convex polytope.
    """

    def __init__(self):
        pass

    def __call__(self, constr_para):
        """The functor gets the constraint parameter and generates the
        corresponding vertices representation.

        Args:
            constr_para (obj): Torch tensor for the constraint
                parameter with shape (N, n_constr_para). We assume that the
                constraint parameter consists of concatenated vertices
                coordinates (x0,y0, ..., xN,yN) of the convex polytope.
        Returns:
            v_polys (obj): Vertice representation of the constraint parameter.
                The output dimensions: (x_lm, y_lm)
        """

        #2-d polytope for landmark within triangle constraint
        #shape (N, n_vertices, dim_vertices)
        #dim_vertices: x, y
        n_vertices = int(constr_para.shape[1] / 2)
        poly_2d = constr_para.new(constr_para.shape[0], n_vertices, 2)
        
        for v in range(n_vertices):
            poly_2d[:, v, 0] = constr_para[:, 2*v]
            poly_2d[:, v, 1] = constr_para[:, 2*v+1]

        v_polys = [[poly_2d,]]

        return v_polys


class ConvexPoly(nn.Module):
    """ This nn.Module maps a latent vector in R^N to an output region defined
    by a convex polytope with a fixed number of vertices. The shape of this
    convex polytope is passed as additional input to this module.
    """
    def __init__(self, convex_poly_format):
        """Informs the instance about the expected format of convex polytopes.

        Args:
            convex_poly_format (tuple): Tuple (n_v, dim_v) with two entries for
                the number of vertices n_v of and for the dimension of the
                convex polytope dim_v. 
        """
        super(ConvexPoly, self).__init__()
        self.convex_poly_format = convex_poly_format
        self.dim_z = self.convex_poly_format2dim_z(convex_poly_format)
        self.dim_out = self.convex_poly_format2dim_out(convex_poly_format)

    @staticmethod
    def convex_poly_format2dim_z(convex_poly_format):
        """Extracts the required dimensions for the intermediate variable z from
        convex_poly_format.
        
        Args:
            convex_poly_format (tuple): Tuples (n_v, dim_v) with number of 
                vertices and number of dimensions of convex polytope.
        Returns:
            dim_out (int): Number of output dimensions for given 
                convex_poly_format.
        """
        return convex_poly_format[0]

    @staticmethod
    def convex_poly_format2dim_out(convex_poly_format):
        """Extracts the number of output dimensions from convex_poly_format.

        Args:
            convex_poly_format (tuple): Tuples (n_v, dim_v) with number of 
                vertices and number of dimensions of convex polytope.
        Returns:
            dim_out (int): Number of output dimensions for given 
                convex_poly_format.
        """
        return convex_poly_format[1]

    @staticmethod
    def v_convex_poly2convex_poly_format(v_convex_poly):
        """Extract the convex_poly_format from vertex representation of convex
        polytope.

        Args:
            v_convex_poly (obj): Torch tensor representing the vertex 
                representation of the convex polytope.
        Returns:
            v_convex_poly_format (tuple): Tuple of the number of vertices and
                the dimension (n_v, dim_v).
        """
        n_v = v_convex_poly.shape[1]
        dim_v = v_convex_poly.shape[2]
        return (n_v, dim_v)

    def forward(self, z, v_convex_poly):
        """
        Args:
            z (obj): Torch tensor with latent representation. Shape 
                (N, n_v)
            v_convex_poly (obj): Pytorch tensor with convex polytope 
                representation. Shape (N, n_v, dim_v).
        Returns:
            out (obj): Torch tensor with shape (N, dim_v). Each output is 
                within convex polytope specified by v_convex_poly.
        """
        #check convex_poly_format
        obs_convex_poly_format = self.v_convex_poly2convex_poly_format(v_convex_poly)
        if not self.convex_poly_format == obs_convex_poly_format:
            raise TypeError('Expected convex_poly_format does not match \
                    observed one.')

        if not z.shape[1] == self.dim_z:
            raise TypeError('Expected {z_dim} dimensions for latent \
                    representation but observed {z_dim_nn}.'.format(
                        z_dim = self.dim_z,
                        z_dim_nn = z.shape[1])
                    )

        #shape of z: (N, n_v)
        p = F.softmax(z)
        #change shape to: (N, 1, n_v)
        p = p.view(
                p.shape[0],
                -1,
                p.shape[1]
                )
        #p(N, 1, n_v) * v_convex_poly (N, n_v, dim_v) 
        #= out (N, 1, dim_v)
        out = torch.bmm(p, v_convex_poly)
        #out (N, dim_v)
        out = out.view(out.shape[0], -1)
        
        #check output dimensions
        if not out.shape[1] == self.dim_out:
            raise TypeError('Expected {dim_out} output dimensions but observed \
                    {dim_out_nn}.'.format(
                        dim_out = self.dim_out,
                        dim_out_nn = out.shape[1])
                    )
        return out

class Poly(nn.Module):
    """This nn.Module maps an intermediate variable z to an output region in
    form of a polytope which is defined by several convex polytopes.

    The shape of the non convex polytope is passed by a number of convex 
    polytopes as additional input.
    Note: The functionality for non-convex polytopes is not considered in the
    paper and focus of future research.
    """
    def __init__(self, poly_format):
        """Generates information about the required dimension of the
        intermediate variable z and the output dimension via poly_format.

        Args:
            poly_format (list): List of tuples (n_v, dim_v) for each convex 
                polytope which is part of the total polytope.
        """
        super(Poly, self).__init__()
        #ConvexPoly nn.Module is used for several polytopes
        self.convex_polys = []
        for convex_poly_format in poly_format:
            self.convex_polys.append(ConvexPoly(convex_poly_format))
        self.poly_format = poly_format
        #number of convex polytopes 
        self.n_convex_poly = self.poly_format2n_convex_poly(poly_format)
        #expected dimension of the latent representation
        self.dim_z = self.poly_format2dim_z(poly_format)
        #expected dimensions of the output
        self.dim_out = self.poly_format2dim_out(poly_format)
        if self.n_convex_poly == 0:
            raise TypeError('Polytope must be constructed by at least one \
                    convex polytope.')
        
    @staticmethod
    def poly_format2dim_z(poly_format):
        """Extracts the required dimensions of the intermediate variable z from
        poly_format.
        
        Args:
            poly_format (list): List of tuples (n_v, dim_v) for each convex 
                polytope which is part of the total polytope.
        Returns:
            dim_z (int): Number of latent vector dimensions for given 
                poly_format.
        """
        #dimension of the latent representation
        dim_z = 0
        for convex_poly_format in poly_format:
            dim_z += ConvexPoly.convex_poly_format2dim_z(convex_poly_format)
        #if the polytope is described by more than one convex polytope a 
        #softmax is added and
        n_convex_poly = Poly.poly_format2n_convex_poly(poly_format)
        if n_convex_poly > 1:
            self.dim_z += n_convex_poly

        return dim_z

    @staticmethod
    def poly_format2dim_out(poly_format):
        """Extracts the number of output dimensions from poly_format.

        Args:
            poly_format (list): List of tuples (n_v, dim_v) for each convex 
                polytope which is part of the total polytope.
        Returns:
            dim_out (int): Number of output dimensions for given poly_format.
        """
        #dimensions of the output
        dim_out = 0
        for convex_poly_format in poly_format:
            dim_out += ConvexPoly.convex_poly_format2dim_out(convex_poly_format)
        #if the polytope is described by more than one convex polytope a 
        #softmax is added and 
        n_convex_poly = Poly.poly_format2n_convex_poly(poly_format)
        if n_convex_poly > 1:
            self.dim_out += n_convex_poly
        
        return dim_out
    
    @staticmethod
    def poly_format2n_convex_poly(poly_format):
        """Extracts the number of convex polytopes from poly_format.

        Args:
            poly_format (list): List of tuples (n_v, dim_v) for each convex 
                polytope which is part of the total polytope.
        Returns:
            n_convex_poly (int): Number of convex polytopes for given 
                poly_format.
        """
        return len(poly_format)

    @staticmethod
    def v_poly2poly_format(v_poly):
        """Extract the polytope format from vertex description of several 
        convex polytopes.

        Args:
            v_poly (list): List of convex polytope vertice representations 
                which describe an eventually non-convex polytope. v_poly
                consists of torch tensor elements.
        Returns:
            poly_format (list): List of tuples (N, n_v, dim_v) of convex
                polytopes.
        """
        poly_format = []
        for v_convex_poly in v_poly:
            convex_poly_format = ConvexPoly.v_convex_poly2convex_poly_format(v_convex_poly)
            poly_format.append(convex_poly_format)
        return poly_format



    def forward(self, z, v_poly):
        """
        Args:
            z (obj): Torch tensor with intermediate representation with shape 
                (N,dim_z).
                dim_z = sum of number of vertices of convex polytopes + number 
                of convex polytopes when this number is at least two.
            v_poly (list): List of Torch tensors (N, n_v, dim_v) representing 
                convex polytopes. [ (), (), ...]
        Returns:
            out (obj): Torch tensor of shape (N, n_out). n_out = sum of 
                dimensions of each convex polytope + number of convex polytopes 
                when this number is at least two. Format is 
                (p_1, ..., p_K, y_1_1, .. y_1_L, ..., y_k_1, .. y_K_M)
                p_1, ... p_K: probabilities for each convex polytope when 
                number of convex polytopes is greater equal 2. Otherwise these
                probabilities are discarded. y_i_j: Coordinate j within convex 
                polytope i.
        """
        if not z.shape[1] == self.dim_z:
            raise TypeError('Dimension of latent representation in nn is \
                    {dim_z_nn} and required for polytope is {dim_z_poly}. \
                    They should be equal.'.format(
                        dim_z_nn = z.shape[1],
                        dim_z_poly = self.dim_z)
                    )
        
        if not self.poly_format == self.v_poly2poly_format(v_poly):
            raise TypeError('Expectet poly_format, i.e. number of convex \
                    polytopes, number of vertices and their dimensions, does \
                    not match with passed vertices representation of \
                    polytope.')

        #add probabilities for each convex polytope to the output when number
        #of them is greater or equal two.
        out = z.new(z.shape[0], self.dim_out)
        z_current_idx = 0
        out_current_idx = 0
        if self.n_convex_poly > 1:
            #shape: (N, n_convex_poly)
            out = F.softmax(z[:,0:self.n_convex_poly])
            z_current_idx = self.n_convex_poly
            out_current_idx = self.n_convex_poly
        
        for i, convex_poly in enumerate(self.convex_polys):
            v_convex_poly = v_poly[i]
            out[:, out_current_idx: out_current_idx + convex_poly.dim_out] = \
                    convex_poly(
                        z[:, z_current_idx: z_current_idx + convex_poly.dim_z],
                        v_convex_poly)
            z_current_idx += convex_poly.dim_z
            out_current_idx += convex_poly.dim_out

        return out



def opts2polys(opts):
    """Creates Polys nn.Modules by calling its constructor with options from 
    opts.

    Args:
        opts (obj): Namespace object with options.

    Returns:
        polys (obj): Instantiated Polys nn.Module.
    """
    #e.g. poly_formats = [[(2,1)],[(3,2)],[(5,3)]]
    #opts.polys_convex_polys_v_n = 2, 3, 5
    #opts.polys_convex_polys_v_dim = 1, 2, 3
    #opts.polys_output_parts = 1, 1, 1
    if not len(opts.polys_convex_polys_v_n) == len(opts.polys_convex_polys_v_dim):
        raise TypeError('Number of list elements in opts.polys_convex_polys_v_n \
                and opts.polys_convex_polys_v_dim must be equal but is not.')
    if not len(opts.polys_convex_polys_v_n) == len(opts.polys_output_parts):
        raise TypeError('Number of list elements in opts.polys_convex_polys_v_n \
                and opts.polys_output_parts must be equal but is not.')
    poly_formats = []
    current_idx = 0
    for n_convex_polys in opts.polys_output_parts:
        poly_format = []
        for i_convex_poly in range(n_convex_polys):
            v_n = opts.polys_convex_polys_v_n[i_convex_poly + current_idx]
            v_dim = opts.polys_convex_polys_v_dim[i_convex_poly + current_idx]
            poly_format.append((v_n, v_dim))
            current_idx += n_convex_polys
        poly_formats.append(poly_format)

    print('Polys loaded as constraint-guard layer.')

    return Polys(poly_formats)


class Polys(nn.Module):
    """Constraint-guard layer to constrain output-parts to polytopes. Currently
    we consider only convex polytopes.

    Different output parts are constrained to different polytopes 
    independently. These polytopes are passed to this functor as additional
    input in the vertices format v_polys.
    """
    def __init__(self, poly_formats):
        """Generates information about required dimensions for intermediate
        representation and output dimensions.

        Args:
            poly_formats (list): List of poly_format objects (see Poly) for 
                the different polytopes of the output parts.
        """
        super(Polys, self).__init__()
        self.poly_formats = poly_formats
        self.polys = []
        for poly_format in self.poly_formats:
            poly = Poly(poly_format)
            self.polys.append(poly)

        #expected number of dimensions for intermediate variable z
        self.dim_z = self.poly_formats2dim_z(poly_formats)
        #expected number of ouput dimensions
        self.dim_out = self.poly_formats2dim_out(poly_formats)

    @staticmethod
    def poly_formats2dim_z(poly_formats):
        """Extracts the number of required dimensions of the intermediate
        variable z from poly_formats.
        
        Args:
            poly_formats (list): List of poly_format objects (see Poly) for 
                the different polytopes of the output parts.
        Returns:
            dim_z (int): Number of required dimensions of intermediate variable.
        """
        dim_z = 0
        for poly_format in poly_formats:
            dim_z += Poly.poly_format2dim_z(poly_format)
        return dim_z

    @staticmethod
    def poly_formats2dim_out(poly_formats):
        """Extracts the number of output dimensions from poly_formats.

        Args:
            poly_formats (list): List of poly_format objects (see Poly) for 
                the different polytopes of the output parts.
        Returns:
            dim_out (int): Number of output dimensions for given poly_format.
        """
        dim_out = 0
        for poly_format in poly_formats:
            dim_out += Poly.poly_format2dim_out(poly_format)
        return dim_out

    @staticmethod
    def v_polys2poly_formats(v_polys):
        """Extract the polytope formats from vertex representation of several 
        polytopes.

        Args:
            v_polys (list): List of polytope vertex representations. v_polys
                consists of list elements.
        Returns:
            poly_formats (list): List of poly_format elements.
        """
        poly_formats = []
        for v_poly in v_polys:
            poly_format = Poly.v_poly2poly_format(v_poly)
            poly_formats.append(poly_format)
        return poly_formats

    def forward(self, z, v_polys):
        """
        Args:
            z (obj): Torch tensor with latent representation. Shape (N, n_z).
            v_polys (list): List with polytope description for different output
                parts. 
        Returns:
            out (obj): Torch tensor with 
        """
        #check correct shape of latent representation
        if not z.shape[1] == self.dim_z:
            raise TypeError('Dimension of intermediate representation z is \
                    {dim_z_nn}, but {dim_z} was expected.'.format(
                        dim_z_nn = z.shape[1],
                        dim_z = self.dim_z)
                    )

        #check if v_polys maps with expected poly_formats
        if not self.v_polys2poly_formats(v_polys) == self.poly_formats:
            raise TypeError('Expected format of v_polys given by poly_formats \
                    does not match observed format inferred from v_polys. \n \
                            poly_formats: {poly_formats} \n \
                            v_polys2poly_formats(v_polys): {polys2polys_format}'.format(
                poly_formats=self.poly_formats,
                polys2polys_format=self.v_polys2poly_formats(v_polys)
            ))

        #output tensor with required dimension
        y = z.new(z.shape[0], self.dim_out)
        
        z_current_idx = 0
        y_current_idx = 0
        for i, poly in enumerate(self.polys):
            dim_z_i = poly.dim_z
            dim_y_i = poly.dim_out
            v_poly = v_polys[i]
            y[:, y_current_idx: y_current_idx + dim_y_i] = \
                    poly(z[:, z_current_idx: z_current_idx + dim_z_i], v_poly)
            z_current_idx += dim_z_i
            y_current_idx += dim_y_i
        
        return y
