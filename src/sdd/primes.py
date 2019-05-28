#!/usr/bin/env python

"""
compile the prime implicants of an SDD/BDD
"""

from collections import defaultdict
import sdd
import models

cache_hits = 0

########################################
# PRIME IMPLICANTS
########################################

# OLD ONE
def _primes_one(alpha,variables,cache,cache_dummy,pmgr,mgr):
    if len(variables) == 0:
        if sdd.sdd_node_is_true(alpha): return sdd.sdd_manager_true(pmgr)
        if sdd.sdd_node_is_false(alpha): return sdd.sdd_manager_false(pmgr)
    #add cases for true/false

    key = (len(variables),sdd.sdd_id(alpha))
    if key in cache:
        global cache_hits
        cache_hits += 1
        #if cache_hits % 1000 == 0: print "cache-hits-update:", cache_hits
        return cache[key]

    var,remaining = variables[0],variables[1:]
    beta2 = sdd.sdd_forall(var,alpha,mgr)
    gamma2 = _primes_one(beta2,remaining,cache,cache_dummy,pmgr,mgr)
    gamma9 = gamma2
    pvar = 3*(var-1)+1
    kappa2 = sdd.sdd_manager_literal(-pvar,pmgr)
    gamma2 = sdd.sdd_conjoin(gamma2,kappa2,pmgr)

    beta0 = sdd.sdd_condition(-var,alpha,mgr)
    gamma0 = _primes_one(beta0,remaining,cache,cache_dummy,pmgr,mgr)
    gamma0 = sdd.sdd_conjoin(gamma0,sdd.sdd_negate(gamma9,pmgr),pmgr)
    kappa0 = sdd.sdd_conjoin(sdd.sdd_manager_literal(-(pvar+1),pmgr),
                             sdd.sdd_manager_literal( (pvar+2),pmgr),pmgr)
    kappa0 = sdd.sdd_conjoin(kappa0,sdd.sdd_manager_literal(pvar,pmgr),pmgr)
    gamma0 = sdd.sdd_conjoin(gamma0,kappa0,pmgr)
    #gamma0 = sdd.sdd_conjoin(gamma0,sdd.sdd_negate(gamma9,pmgr),pmgr)

    beta1 = sdd.sdd_condition(var,alpha,mgr)
    gamma1 = _primes_one(beta1,remaining,cache,cache_dummy,pmgr,mgr)
    gamma1 = sdd.sdd_conjoin(gamma1,sdd.sdd_negate(gamma9,pmgr),pmgr)
    kappa1 = sdd.sdd_conjoin(sdd.sdd_manager_literal( (pvar+1),pmgr),
                             sdd.sdd_manager_literal(-(pvar+2),pmgr),pmgr)
    kappa1 = sdd.sdd_conjoin(kappa1,sdd.sdd_manager_literal(pvar,pmgr),pmgr)
    gamma1 = sdd.sdd_conjoin(gamma1,kappa1,pmgr)
    #gamma1 = sdd.sdd_conjoin(gamma1,sdd.sdd_negate(gamma9,pmgr),pmgr)

    gamma = sdd.sdd_disjoin(gamma0,gamma1,pmgr)
    gamma = sdd.sdd_disjoin(gamma,gamma2,pmgr)

    cache[key] = gamma
    return gamma

def _remove_dummies(alpha,var_count,pmgr):
    for var in xrange(1,var_count+1):
        var = 3*(var-1)+1
        beta = sdd.sdd_manager_literal(-var,pmgr)
        gamma = sdd.sdd_disjoin(sdd.sdd_manager_literal(var+1,pmgr),
                                sdd.sdd_manager_literal(var+2,pmgr),pmgr)
        beta = sdd.sdd_conjoin(beta,gamma,pmgr)
        alpha = sdd.sdd_conjoin(alpha,sdd.sdd_negate(beta,pmgr),pmgr)
    return alpha

def prime_to_term(prime,mgr):
    """converts a prime from the IP-SDD to a term in the original manager
    
    assumes prime is an IP-model (dict from var to value)"""
    var_count = sdd.sdd_manager_var_count(mgr)
    term = sdd.sdd_manager_true(mgr)
    for var in xrange(1,var_count+1):
        pvar = 3*(var-1)+1
        if prime[pvar] == 0: continue
        val = prime[pvar+1]
        lit = var if val == 1 else -var
        lit = sdd.sdd_manager_literal(lit,mgr)
        term = sdd.sdd_conjoin(term,lit,mgr)
    return term

def prime_to_dict(prime,var_count):
    """converts a prime from the IP-SDD to a term in the original manager
    
    assumes prime is an IP-model (dict from var to value)"""
    term = {}
    for var in xrange(1,var_count+1):
        pvar = 3*(var-1)+1
        if prime[pvar] == 0: continue
        val = prime[pvar+1]
        term[var] = val
    return term

########################################
# MORE PRIME IMPLICANTS (alt algorithm)
########################################

def _sdd_used_neg(var,pmgr):
    pvar = 3*(var-1)+1
    kappa = sdd.sdd_conjoin(sdd.sdd_manager_literal(-(pvar+1),pmgr),
                            sdd.sdd_manager_literal( (pvar+2),pmgr),pmgr)
    kappa = sdd.sdd_conjoin(kappa,sdd.sdd_manager_literal(pvar,pmgr),pmgr)
    return kappa

def _sdd_used_pos(var,pmgr):
    pvar = 3*(var-1)+1
    kappa = sdd.sdd_conjoin(sdd.sdd_manager_literal( (pvar+1),pmgr),
                            sdd.sdd_manager_literal(-(pvar+2),pmgr),pmgr)
    kappa = sdd.sdd_conjoin(kappa,sdd.sdd_manager_literal(pvar,pmgr),pmgr)
    return kappa

def _sdd_unused(var,pmgr):
    pvar = 3*(var-1)+1
    kappa = sdd.sdd_conjoin(sdd.sdd_manager_literal(-(pvar+1),pmgr),
                            sdd.sdd_manager_literal(-(pvar+2),pmgr),pmgr)
    kappa = sdd.sdd_conjoin(kappa,sdd.sdd_manager_literal(-pvar,pmgr),pmgr)
    return kappa

def _keep_imp(beta,alpha,variables,cache1,cache2,pmgr,mgr):
    #if len(variables) == 0:
    if sdd.sdd_node_is_false(beta): return sdd.sdd_manager_false(pmgr)
    if sdd.sdd_node_is_false(alpha): return sdd.sdd_manager_false(pmgr)
    if sdd.sdd_node_is_true(alpha): return beta

    key = (len(variables),sdd.sdd_id(alpha),sdd.sdd_id(beta))
    if key in cache2:
        global cache_hits
        cache_hits += 1
        #if cache_hits % 1000 == 0: print "cache-hits-update:", cache_hits
        return cache2[key]

    var,remaining = variables[0],variables[1:]
    pvar = 3*(var-1)+1
    alpha0 = sdd.sdd_condition(-var,alpha,mgr)
    alpha1 = sdd.sdd_condition(var,alpha,mgr)
    beta0 = sdd.sdd_condition(pvar,beta,pmgr)
    beta0 = sdd.sdd_condition(-(pvar+1),beta0,pmgr)
    #beta0 = sdd.sdd_condition( (pvar+2),beta0,pmgr)
    beta1 = sdd.sdd_condition(pvar,beta,pmgr)
    beta1 = sdd.sdd_condition((pvar+1),beta1,pmgr)
    #beta1 = sdd.sdd_condition(-(pvar+2),beta1,pmgr)
    betad = sdd.sdd_condition(-pvar,beta,pmgr)

    P = _keep_imp(betad,alpha0,remaining,cache1,cache2,pmgr,mgr)
    Q = _keep_imp(betad,alpha1,remaining,cache1,cache2,pmgr,mgr)
    R0 = _keep_imp(beta0,alpha0,remaining,cache1,cache2,pmgr,mgr)
    R1 = _keep_imp(beta1,alpha1,remaining,cache1,cache2,pmgr,mgr)

    gamma = sdd.sdd_conjoin(P,Q,pmgr)
    gamma = sdd.sdd_conjoin(_sdd_unused(var,pmgr),gamma,pmgr)
    kappa = sdd.sdd_conjoin(_sdd_used_neg(var,pmgr),R0,pmgr)
    gamma = sdd.sdd_disjoin(gamma,kappa,pmgr)
    kappa = sdd.sdd_conjoin(_sdd_used_pos(var,pmgr),R1,pmgr)
    gamma = sdd.sdd_disjoin(gamma,kappa,pmgr)

    cache2[key] = gamma
    return gamma

# NEW ONE
def _primes_two(alpha,variables,cache1,cache2,pmgr,mgr):
    if len(variables) == 0:
        if sdd.sdd_node_is_false(alpha): return sdd.sdd_manager_false(pmgr)
        if sdd.sdd_node_is_true(alpha): return sdd.sdd_manager_true(pmgr)

    key = (len(variables),sdd.sdd_id(alpha))
    if key in cache1:
        global cache_hits
        cache_hits += 1
        if cache_hits % 1000 == 0: print "cache-hits-update:", cache_hits
        return cache1[key]

    var,remaining = variables[0],variables[1:]
    alpha0 = sdd.sdd_condition(-var,alpha,mgr)
    alpha1 = sdd.sdd_condition(var,alpha,mgr)
    primes0 = _primes_two(alpha0,remaining,cache1,cache2,pmgr,mgr)
    primes1 = _primes_two(alpha1,remaining,cache1,cache2,pmgr,mgr)
    qrimes0 = _keep_imp(primes0,alpha1,remaining,cache1,cache2,pmgr,mgr)
    qrimes1 = _keep_imp(primes1,alpha0,remaining,cache1,cache2,pmgr,mgr)

    gamma = sdd.sdd_disjoin(qrimes0,qrimes1,pmgr)
    gamma = sdd.sdd_conjoin(_sdd_unused(var,pmgr),gamma,pmgr)
    kappa = sdd.sdd_conjoin(primes0,sdd.sdd_negate(qrimes0,pmgr),pmgr)
    kappa = sdd.sdd_conjoin(kappa,_sdd_used_neg(var,pmgr),pmgr)
    gamma = sdd.sdd_disjoin(gamma,kappa,pmgr)
    kappa = sdd.sdd_conjoin(primes1,sdd.sdd_negate(qrimes1,pmgr),pmgr)
    kappa = sdd.sdd_conjoin(kappa,_sdd_used_pos(var,pmgr),pmgr)
    gamma = sdd.sdd_disjoin(gamma,kappa,pmgr)

    cache1[key] = gamma
    return gamma

# CHANGE DEFAULT PRIMES IMPLEMENTATION HERE:
def primes(alpha,mgr,primes_f=_primes_two):
    var_count = sdd.sdd_manager_var_count(mgr)
    primes_var_count = 3*var_count
    primes_vtree = sdd.sdd_vtree_new(primes_var_count,"balanced")
    primes_mgr = sdd.sdd_manager_new(primes_vtree)
    variables = range(1,var_count+1)
    cache1,cache2 = {},{}
    kappa = primes_f(alpha,variables,cache1,cache2,primes_mgr,mgr)
    kappa = _remove_dummies(kappa,var_count,primes_mgr)
    return kappa,primes_mgr

########################################
# ENUMERATION BY SIZE
########################################

def enumerate_primes(primes,pmgr,var_count):
    pvtree = sdd.sdd_manager_vtree(pmgr)
    while not sdd.sdd_node_is_false(primes):
        mincard = sdd.sdd_global_minimize_cardinality(primes,pmgr)
        for model in models.models(mincard,pvtree):
            term = prime_to_dict(model,var_count)
            yield term
        primes = sdd.sdd_conjoin(primes,sdd.sdd_negate(mincard,pmgr),pmgr)

def primes_by_length(primes,pmgr,var_count):
    by_length = defaultdict(list)
    pvtree = sdd.sdd_manager_vtree(pmgr)
    for model in models.models(primes,pvtree):
        term = prime_to_dict(model,var_count)
        by_length[len(term)].append(term)
    return by_length

########################################
# COMPATIBLE PRIME IMPLICANTS
########################################

def compatible_primes(alpha,inst,mgr,primes_mgr=None):
    if primes_mgr is None:
        beta,pmgr = primes(alpha,mgr)
    else:
        beta,pmgr = primes_mgr
    asdf = beta
    for i,val in enumerate(inst):
        var = i+1
        pvar = 3*(var-1)+1
        lit = (pvar+1) if val == 1 else -(pvar+1)
        gamma = sdd.sdd_conjoin(sdd.sdd_manager_literal(pvar,pmgr),
                                sdd.sdd_manager_literal(lit,pmgr),pmgr)
        gamma = sdd.sdd_disjoin(gamma,
                                sdd.sdd_manager_literal(-pvar,pmgr),pmgr)
        beta = sdd.sdd_conjoin(beta,gamma,pmgr)
    return beta,pmgr

########################################
# TESTING
########################################

def _is_prime(prime,f,mgr):
    var_count = sdd.sdd_manager_var_count(mgr)
    term = prime_to_dict(prime,var_count)
    for var in term:
        alpha = f
        for varp in term:
            if varp == var: continue
            lit = varp if term[varp] == 1 else -varp
            alpha = sdd.sdd_condition(lit,alpha,mgr)
        if sdd.sdd_node_is_true(alpha):
            return False
    return True

def _sanity_check(f,mgr,g,pmgr):
    """f is original function and g is its prime implicants"""

    alpha = sdd.sdd_manager_false(mgr)
    pvtree = sdd.sdd_manager_vtree(pmgr)
    for prime in models.models(g,pvtree):
        term = prime_to_term(prime,mgr)
        beta = sdd.sdd_conjoin(term,f,mgr)
        assert term == beta
        assert _is_prime(prime,f,mgr)
        alpha = sdd.sdd_disjoin(alpha,term,mgr)
    mc1 = sdd.sdd_global_model_count(f,mgr)
    mc2 = sdd.sdd_global_model_count(alpha,mgr)
    print "mc-check:", mc1, mc2, ("ok" if mc1 == mc2 else "NOT OK")
    assert mc1 == mc2
    assert alpha == f

def test():
    var_count = 4
    vtree = sdd.sdd_vtree_new(var_count,"balanced")
    mgr = sdd.sdd_manager_new(vtree)
    
    # A v B
    alpha = sdd.sdd_disjoin(sdd.sdd_manager_literal(1,mgr),
                            sdd.sdd_manager_literal(2,mgr),mgr)
    beta  = sdd.sdd_conjoin(sdd.sdd_manager_literal(-3,mgr),
                            sdd.sdd_manager_literal(-4,mgr),mgr)
    # A v B v ( ~C ^ ~D )
    alpha = sdd.sdd_disjoin(alpha,beta,mgr)
    
    beta,pmgr = primes(alpha,mgr)
    _sanity_check(alpha,mgr,beta,pmgr)
    pvtree = sdd.sdd_manager_vtree(pmgr)

    import models
    #beta2 = sdd.sdd_global_minimize_cardinality(beta,pmgr)
    beta2 = beta
    for model in models.models(beta2,pvtree):
        print models.str_model(model)

    global cache_hits
    print "cache-hits:", cache_hits

    print "all-ones"
    beta,pmgr = compatible_primes(alpha,[1,1,1,1],mgr)
    pvtree = sdd.sdd_manager_vtree(pmgr)
    for model in models.models(beta,pvtree):
        print models.str_model(model)

    print "all-zeros"
    beta,pmgr = compatible_primes(alpha,[0,0,0,0],mgr)
    pvtree = sdd.sdd_manager_vtree(pmgr)
    for model in models.models(beta,pvtree):
        print models.str_model(model)

    print "blah"
    beta,pmgr = compatible_primes(alpha,[1,0,1,0],mgr)
    pvtree = sdd.sdd_manager_vtree(pmgr)
    for model in models.models(beta,pvtree):
        print models.str_model(model)

    print "dead-nodes:", sdd.sdd_manager_dead_count(mgr)
    print "dead-nodes:", sdd.sdd_manager_dead_count(pmgr)

def test_andy():
    var_count = 3
    vtree = sdd.sdd_vtree_new(var_count,"balanced")
    mgr = sdd.sdd_manager_new(vtree)

    # 100, 101, 111, 001, 011
    alpha = sdd.sdd_manager_false(mgr)
    beta = sdd.sdd_conjoin(sdd.sdd_manager_literal(1,mgr),
                           sdd.sdd_manager_literal(-2,mgr),mgr)
    beta = sdd.sdd_conjoin(sdd.sdd_manager_literal(-3,mgr),beta,mgr)
    alpha = sdd.sdd_disjoin(alpha,beta,mgr)
    beta = sdd.sdd_conjoin(sdd.sdd_manager_literal(1,mgr),
                           sdd.sdd_manager_literal(-2,mgr),mgr)
    beta = sdd.sdd_conjoin(sdd.sdd_manager_literal(3,mgr),beta,mgr)
    alpha = sdd.sdd_disjoin(alpha,beta,mgr)
    beta = sdd.sdd_conjoin(sdd.sdd_manager_literal(1,mgr),
                           sdd.sdd_manager_literal(2,mgr),mgr)
    beta = sdd.sdd_conjoin(sdd.sdd_manager_literal(3,mgr),beta,mgr)
    alpha = sdd.sdd_disjoin(alpha,beta,mgr)
    beta = sdd.sdd_conjoin(sdd.sdd_manager_literal(-1,mgr),
                           sdd.sdd_manager_literal(-2,mgr),mgr)
    beta = sdd.sdd_conjoin(sdd.sdd_manager_literal(3,mgr),beta,mgr)
    alpha = sdd.sdd_disjoin(alpha,beta,mgr)
    beta = sdd.sdd_conjoin(sdd.sdd_manager_literal(-1,mgr),
                           sdd.sdd_manager_literal(2,mgr),mgr)
    beta = sdd.sdd_conjoin(sdd.sdd_manager_literal(3,mgr),beta,mgr)
    alpha = sdd.sdd_disjoin(alpha,beta,mgr)
    
    beta,pmgr = primes(alpha,mgr)
    _sanity_check(alpha,mgr,beta,pmgr)
    vtree = sdd.sdd_manager_vtree(mgr)
    pvtree = sdd.sdd_manager_vtree(pmgr)

    import models
    for model in models.models(alpha,vtree):
        print models.str_model(model)

    for model in models.models(beta,pvtree):
        print models.str_model(model)

    print "dead-nodes:", sdd.sdd_manager_dead_count(mgr)
    print "dead-nodes:", sdd.sdd_manager_dead_count(pmgr)

def test_admission():
    var_count = 4
    vtree = sdd.sdd_vtree_new(var_count,"balanced")
    mgr = sdd.sdd_manager_new(vtree)

    # WFEG
    # ( w ^ g )
    alpha = sdd.sdd_conjoin(sdd.sdd_manager_literal(1,mgr),
                            sdd.sdd_manager_literal(4,mgr),mgr)
    # ( w ^ f ^ e )
    beta = sdd.sdd_conjoin(sdd.sdd_manager_literal(1,mgr),
                           sdd.sdd_manager_literal(2,mgr),mgr)
    beta = sdd.sdd_conjoin(beta,
                           sdd.sdd_manager_literal(3,mgr),mgr)
    # ( f ^ e ^ g )
    gamma = sdd.sdd_conjoin(sdd.sdd_manager_literal(2,mgr),
                            sdd.sdd_manager_literal(3,mgr),mgr)
    gamma = sdd.sdd_conjoin(gamma,
                            sdd.sdd_manager_literal(4,mgr),mgr)
    alpha = sdd.sdd_disjoin(alpha,beta,mgr)
    alpha = sdd.sdd_disjoin(alpha,gamma,mgr)

    alpha = sdd.sdd_negate(alpha,mgr)
    beta,pmgr = primes(alpha,mgr)
    _sanity_check(alpha,mgr,beta,pmgr)
    vtree = sdd.sdd_manager_vtree(mgr)
    pvtree = sdd.sdd_manager_vtree(pmgr)

    import models
    for model in models.models(alpha,vtree):
        print models.str_model(model)

    for model in models.models(beta,pvtree):
        print models.str_model(model)

    for model in models.models(alpha,vtree):
        print "==", models.str_model(model)
        model_list = [ model[var] for var in sorted(model.keys()) ]
        gamma,pmgr = compatible_primes(alpha,model_list,mgr,primes_mgr=(beta,pmgr))
        pvtree = sdd.sdd_manager_vtree(pmgr)
        for prime_model in models.models(gamma,pvtree):
            print models.str_model(prime_model)
            term = prime_to_dict(prime_model,var_count)
            print " ".join([ ("*" if var not in term else "+" if term[var] == 1 else "-") for var in xrange(1,var_count+1) ])

    print "dead-nodes:", sdd.sdd_manager_dead_count(mgr)
    print "dead-nodes:", sdd.sdd_manager_dead_count(pmgr)
