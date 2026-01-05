"""
    Physics

Game-theoretic environment rules.
Defines payoff matrices for various social dilemmas.

CRITICAL: The environment is LABEL-BLIND - outcomes depend ONLY on behaviors,
never on agent labels. This ensures any label-based patterns that emerge
are purely cognitive constructions.
"""
module Physics

export GameType, PrisonersDilemma, StagHunt, Harmony,
       resolve_interaction, get_payoff_matrix, game_name,
       PAYOFF_MATRICES

"""
    GameType

Abstract type for different game-theoretic interactions.
Each game type defines a different payoff structure for the 2x2 game.
"""
abstract type GameType end

"""
    PrisonersDilemma <: GameType

Classic Prisoner's Dilemma: defection dominates but mutual cooperation is better.

Payoff matrix (row player perspective):
                Cooperate   Defect
    Cooperate      3          0
    Defect         5          1

Properties:
- Defection is dominant strategy
- Mutual cooperation (3,3) Pareto-dominates mutual defection (1,1)
- Creates social dilemma: individual vs collective rationality
"""
struct PrisonersDilemma <: GameType end

"""
    StagHunt <: GameType

Stag Hunt: coordination game where both cooperation and defection are equilibria.

Payoff matrix:
                Cooperate   Defect
    Cooperate      4          0
    Defect         3          3

Properties:
- Two Nash equilibria: (C,C) and (D,D)
- Requires trust/coordination to achieve optimal outcome
- Tests institution's role in enabling coordination
"""
struct StagHunt <: GameType end

"""
    Harmony <: GameType

Harmony game: cooperation is dominant strategy.

Payoff matrix:
                Cooperate   Defect
    Cooperate      4          3
    Defect         3          2

Properties:
- Cooperation is dominant strategy
- No social dilemma
- Used as control condition (institutions should NOT emerge)
"""
struct Harmony <: GameType end

"""
Global dictionary of payoff matrices.
Format: [CC CD; DC DD] where first letter is row player's action.
"""
const PAYOFF_MATRICES = Dict{Type{<:GameType}, Matrix{Float64}}(
    PrisonersDilemma => Float64[3 0; 5 1],  # Classic PD
    StagHunt         => Float64[4 0; 3 3],  # Coordination game
    Harmony          => Float64[4 3; 3 2],  # No dilemma
)

"""
    get_payoff_matrix(game::GameType) -> Matrix{Float64}

Get the payoff matrix for a game type.
Returns 2x2 matrix where rows are own actions [C, D] and columns are opponent actions [C, D].
"""
function get_payoff_matrix(game::GameType)
    return PAYOFF_MATRICES[typeof(game)]
end

"""
    game_name(game::GameType) -> String

Get human-readable name for a game type.
"""
game_name(::PrisonersDilemma) = "Prisoner's Dilemma"
game_name(::StagHunt) = "Stag Hunt"
game_name(::Harmony) = "Harmony Game"

"""
    resolve_interaction(action1, action2, game) -> Tuple{Float64, Float64}

Resolve an interaction between two agents.
Returns (payoff1, payoff2) based on actions and game type.

IMPORTANT: This function is LABEL-BLIND. It only considers actions,
never the labels of the agents. Any correlation between labels and
outcomes must emerge through agent behavior, not environment rules.

# Arguments
- `action1::Bool`: First agent's action (true=cooperate, false=defect)
- `action2::Bool`: Second agent's action
- `game::GameType`: The game being played

# Returns
- `(payoff1, payoff2)`: Payoffs for each agent
"""
function resolve_interaction(action1::Bool, action2::Bool, game::GameType)
    payoffs = get_payoff_matrix(game)

    # Convert actions to indices: cooperate=1, defect=2
    idx1 = action1 ? 1 : 2
    idx2 = action2 ? 1 : 2

    # Look up payoffs
    payoff1 = payoffs[idx1, idx2]  # Row player (agent 1)
    payoff2 = payoffs[idx2, idx1]  # Column player (agent 2)

    return (payoff1, payoff2)
end

"""
    get_best_response(opponent_action, game) -> Bool

Compute best response to opponent's action (assuming rational payoff maximization).
"""
function get_best_response(opponent_action::Bool, game::GameType)
    payoffs = get_payoff_matrix(game)
    col = opponent_action ? 1 : 2

    payoff_cooperate = payoffs[1, col]
    payoff_defect = payoffs[2, col]

    return payoff_cooperate > payoff_defect
end

"""
    is_social_dilemma(game::GameType) -> Bool

Check if the game has a social dilemma structure
(individual optimal differs from collective optimal).
"""
function is_social_dilemma(game::GameType)
    payoffs = get_payoff_matrix(game)

    # Mutual cooperation payoff
    cc = payoffs[1, 1]
    # Mutual defection payoff
    dd = payoffs[2, 2]
    # Exploitation payoff (defect against cooperator)
    dc = payoffs[2, 1]

    # Social dilemma: DC > CC (temptation to defect) but CC > DD (mutual coop better)
    return dc > cc && cc > dd
end

"""
    nash_equilibria(game::GameType) -> Vector{Tuple{Bool, Bool}}

Find pure strategy Nash equilibria of the game.
Returns list of (action1, action2) equilibrium profiles.
"""
function nash_equilibria(game::GameType)
    equilibria = Tuple{Bool, Bool}[]
    payoffs = get_payoff_matrix(game)

    for a1 in [true, false]
        for a2 in [true, false]
            idx1 = a1 ? 1 : 2
            idx2 = a2 ? 1 : 2

            # Check if a1 is best response to a2
            br1 = payoffs[1, idx2] >= payoffs[2, idx2] ? true : false
            # Actually need to handle ties - use >= for both
            is_br1 = (a1 && payoffs[1, idx2] >= payoffs[2, idx2]) ||
                     (!a1 && payoffs[2, idx2] >= payoffs[1, idx2])

            # Check if a2 is best response to a1
            is_br2 = (a2 && payoffs[1, idx1] >= payoffs[2, idx1]) ||
                     (!a2 && payoffs[2, idx1] >= payoffs[1, idx1])

            if is_br1 && is_br2
                push!(equilibria, (a1, a2))
            end
        end
    end

    return equilibria
end

"""
    pareto_optimal(game::GameType) -> Vector{Tuple{Bool, Bool}}

Find Pareto-optimal action profiles.
"""
function pareto_optimal(game::GameType)
    payoffs = get_payoff_matrix(game)
    profiles = [(true, true), (true, false), (false, true), (false, false)]

    function get_payoffs(p)
        idx1 = p[1] ? 1 : 2
        idx2 = p[2] ? 1 : 2
        return (payoffs[idx1, idx2], payoffs[idx2, idx1])
    end

    optimal = Tuple{Bool, Bool}[]

    for p in profiles
        pays_p = get_payoffs(p)
        is_dominated = false

        for q in profiles
            if p == q
                continue
            end
            pays_q = get_payoffs(q)

            # Check if q dominates p
            if pays_q[1] >= pays_p[1] && pays_q[2] >= pays_p[2] &&
               (pays_q[1] > pays_p[1] || pays_q[2] > pays_p[2])
                is_dominated = true
                break
            end
        end

        if !is_dominated
            push!(optimal, p)
        end
    end

    return optimal
end

end # module
