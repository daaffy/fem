# Simple interface example - but with gmsh.

# https://fenicsproject.discourse.group/t/difference-between-meshvaluecollection-and-meshfunction/5219

# .rtc.xml files per topological dimension

# https://github.com/multiphenics/multiphenicsx/blob/main/tutorials/03_lagrange_multipliers/tutorial_lagrange_multipliers_interface.ipynb
# for constructing the .geo file


# 1.
# observe how the traditional way of generating MeshRestrictions works in the interface example
# i.e., which entities are contained in the left subdomain; does it include the interface itself


# 2.
# needs to replicate 1. without SubDomain class (at least without defining inequalities)
# strategy?
# - construct subdomain for input into MeshRestriction
# - define subdomain not via inequalities; but via proximity of dof to the mesh entity (need to see SubDomain.mark,)
# ? https://fenicsproject.org/qa/4000/constructing-subdomains-from-knowing-indices-cells-the-mesh/