// import { Fragment } from "react";
// Can be achieved by just using <> and not importing fragment


// a component cannot return more than 1 element (listgroup can only return listgroup and not h1)
// this can be solved by wrapping the whole thing in a div or fragments
function ListGroup() {
	let items = [
		'New York',
		'Hong Kong', 
		'London',
		'Pennsylvania'
	];
	// items = [];

	return (
		<>	
			<h1>List</h1>
			{ items.length === 0 ? <p>Empty</p> : null}
			<ul className="list-group">
				{
					// If we're retrieving items from an API it's usually a class
					// So we'll list a property (e.g item.id) instead of just item
					items.map((item) => (<li>{item}</li>))
				}
			</ul>
		</>
	);
}


export default ListGroup;
