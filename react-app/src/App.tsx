import ListGroup from "./components/ListGroup";


function App() {
	let items = [
		"New York",
		"Hong Kong", 
		"London",
		"Pennsylvania"
	];	
	return <div><ListGroup items={items} heading={"Cities"}/></div> 
}

export default App;
