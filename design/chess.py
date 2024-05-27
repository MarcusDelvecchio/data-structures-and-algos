import uuid

# features we should add
# client for initializing player and joining a/the network
# GUI for playing the game could create a simple JS app that interacts with the PlayerNetwork/Game API and allows the user to make moves and serves them the board infoz

# features we could add
# game having chat room
# go forward / back 
# adding a historic games to player and allowing them to view / traverse

class PlayerNetwork:

    def __init__(self):
        self.games = {}
        self.players = {}
        self.endedGames = {}

    def addPlayer(self, player_username: Player):
        if player_username not in self.players:
            self.players[player_username] = Player(player_username, self)
        else:
            print('provide a unique username')

    def addGame(self, game_name):
        if game_name in self.games:
            return "choose a differtent name"
        else:
            # note game name has to be unique
            new_game = ChessGame(game_name)
            self.games[game_name] = new_game
            return new_game
    
    def getGames(self):
        return [self.games[game].name for game in self.games]
    
    def getCurrentGames(self):
        return [self.games[game].name for game in self.games if not game.winner] 

    def getUnstartedGames(self):
        return [self.games[game].name for game in self.games if not game.started]
    
    # returns a game object so the Player can ineract with it
    def getGame(self, game_name):
        return self.games[game_name]
    
    # removes the game from active games and adds to completed games
    def endGame(self, game_name):
        if game_name not in self.games:
            # something went wrong
            return
        else:
            game = self.games[game_name]
            del self.games[game_name]
            self.endedGames.append(game)


class Player:
    
    def __init__(self, username, network):
        self.username = username
        self.network = network
        self.currentGame = None
        self.gameHistory = []
        
    def getGames(self):
        print("please choose a game")
        print(self.network.getUnstartedGames())

        # add logic to allow user to pick a game

    def joinGame(self, game_name):
        game = self.network.getGame(game_name)
        game.addPlayer(self)                

    def createGame(self, game_name):
        new_game = self.network.addGame(game_name)
        self.currentGame = new_game

    def makeMove(self, from_row, from_col, to_row, to_col):
        if not self.currentGame: return
        self.currentGame.makeMove(self, from_row, from_col, to_row, to_col)

        if self.currentGame.winner:
            self.gameHistory.append(self.currentGame)
            self.currentGame = None

class ChessGame:

    def __init__(self, game_name):
        self.id = uuid.uuid4()
        self.name = game_name
        self.started = False
        self.board = ChessBoard()
        self.whiteTurn = True
        self.moves = []
        self.winner = None

        # players (assumed that players[0] is white)
        self.players = []
    
    def addPlayer(self, player: Player):
        if len(self.players) != 2:
            self.players.append(player)
        else:
            print('game full')
        
        if len(self.players) == 2:
            # randomize player order
            self.startGame()
    
    def startGame(self):
        # tell the players to make their move            
        self.started = True        

    def makeMove(self, player, from_row, from_col, to_row, to_col):
        if self.whiteTurn and player == self.players[1] or not self.whiteTurn and player == self.players[0]:
            print("not your turn")
            return

        # validate the move
        # ensure the the piece is the same color as the player
        # check if checkmate / stalemate

        # make the move in the board
        self.board.movePiece(from_row, from_col, to_row, to_col)

        # update the move history
        self.moves.append([[from_row, from_col][to_row, to_col]])
    
    def forfeit(self, player: Player):
        self.winner = self.players[1] if player == self.players[0] else self.players[0]
    
    def draw(self, player: Player):
        # add logic for request draw / respond to draw request
        pass


class ChessBoard:

    def __init__(self):
        self.board = [[Box()]*8 for _ in range(8)]

        # setup the board pawns
        for col in range(8):
            self.board[1][col] = Piece(PieceType.pawn, 'black')
            self.board[6][col] = Piece(PieceType.pawn, 'white')

        # setup the board pieces
        for col in range(8):
            if col == 0 or col == 7:
                self.board[0][col] = Piece(PieceType.rook, 'black')
                self.board[7][col] = Piece(PieceType.rook, 'white')
            elif col == 1 or col == 6:
                self.board[0][col] = Piece(PieceType.knight, 'black')
                self.board[7][col] = Piece(PieceType.knight, 'white')
            elif col == 2 or col == 5:
                self.board[0][col] = Piece(PieceType.bishop, 'black')
                self.board[7][col] = Piece(PieceType.bishop, 'white')
            elif col == 3:
                self.board[0][col] = Piece(PieceType.king, 'black')
                self.board[7][col] = Piece(PieceType.queen, 'white')
            elif col == 4:
                self.board[0][col] = Piece(PieceType.queen, 'black')
                self.board[7][col] = Piece(PieceType.king, 'white')
    
    # assumes that the move is already validated
    def movePiece(self, from_row, from_col, to_row, to_col):
        self.board[to_row][to_col] = self.board[from_row][from_col]
        # add logic to mark the piece as captured
        self.board[from_row][from_col] = None


class Box:

    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.piece = None


class Piece:

    def __init__(self, type, color):
        self.type = type
        self.color = color

class PieceType:
    queen = "Q"
    king = "K"
    rook = "r"
    bishop = "b"
    knight = "k"
    pawn = 'p'


# also see https://github.com/tssovi/grokking-the-object-oriented-design-interview/blob/master/object-oriented-design-case-studies/design-chess.md
# but I only followed their requirements (this isn't finished)